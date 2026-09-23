# Usage: <target> must be one of the formats: func_name@file_path:line_num or file_path:line_num
# Fast mode: only analyze the scope of target_funcs-related basic blocks, which is very fast
#       (no need to do build action)
#       python get_targets.py get -k <kernel_dir> -t <target1> <target2> ... -f
# Normal mode: analyze the whole kernel, the one-time build action is quite slow and get action is very fast
#       python get_targets.py build -k <kernel_dir>
#       python get_targets.py get -k <kernel_dir> -t <target1> <target2> ...

import os
import json
import hashlib
import argparse
import subprocess
import re
from time import time
from tqdm import tqdm
import concurrent.futures

try:
    from tree_sitter import Language, Parser
    from tree_sitter_c import language as tree_sitter_c_language
except ImportError:
    Language = None
    Parser = None
    tree_sitter_c_language = None


DEF_SETS = {
    "outdir": "SyzPilot-analysis", # output directory under kernel dir
    "assembly": "vmlinux.asm",
    "allias": "allias.txt",
    "instru": "instrumentations.json",
    "pcs2funcs": "pcs2funcs.json",
}

start_t = time()
SOURCE_FUNCTION_CACHE = {}
FAST_CACHE_VERSION = 'v6'
VMLINUX_IDENTITY_CACHE = {}
TREE_SITTER_C_PARSER = None
TREE_SITTER_TOPLEVEL_CONTAINER_TYPES = {
    'translation_unit',
    'ERROR',
    'preproc_if',
    'preproc_ifdef',
    'preproc_ifndef',
    'preproc_elif',
    'preproc_else',
}
TREE_SITTER_RECOVERED_FUNC_NAME_NODE_TYPES = {
    'identifier',
    'call_expression',
    'function_declarator',
    'pointer_declarator',
    'parenthesized_declarator',
}
TREE_SITTER_RECOVERED_FUNC_HEADER_HINT_TYPES = {
    'primitive_type',
    'sized_type_specifier',
    'type_identifier',
    'struct',
    'union',
    'enum',
    'static',
    'extern',
    'inline',
    'register',
    'volatile',
    'const',
    'signed',
    'unsigned',
    'long',
    'short',
}
TREE_SITTER_RECOVERED_FUNC_HEADER_BARRIER_TYPES = {
    ';',
    '}',
    'compound_statement',
    'function_definition',
    'declaration',
    'expression_statement',
    'return_statement',
    'if_statement',
    'for_statement',
    'while_statement',
    'do_statement',
    'switch_statement',
}

def print_t(msg: str):
    print(f'[{time()-start_t:.2f}s] {msg}')


def write_json(d: dict, fpath: str):
    with open(fpath, 'w') as f:
        json.dump(d, f, indent=4)


def read_json(fpath: str) -> dict:
    with open(fpath, 'r') as f:
        return json.load(f)


def check_command_injection(input_str: str) -> bool:
    """Check if the user controlled string is safe from command injection"""
    # define dangerous characters and patterns
    dangerous_chars = {
        '&', ';', '|', '`', '$', '(', ')', '<', '>', '*', '?', '\\', '\n', '\r'
    }

    # check dangerous characters
    if any(char in input_str for char in dangerous_chars):
        return True

    return False


def get_vmlinux_cache_identity(kernel_dir: str) -> str:
    """Return a content identity for cache invalidation across kernel rebuilds."""
    vmlinux_path = os.path.realpath(os.path.join(kernel_dir, 'vmlinux'))
    if not os.path.isfile(vmlinux_path):
        return 'missing-vmlinux'
    stat = os.stat(vmlinux_path)
    cache_key = (
        vmlinux_path,
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    cached = VMLINUX_IDENTITY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    identity = ''
    try:
        description = subprocess.check_output(
            ['file', '-b', vmlinux_path],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        match = re.search(r'BuildID(?:\[[^]]+\])?=([0-9a-fA-F]+)', description)
        if match:
            identity = f'build-id:{match.group(1).lower()}'
    except (OSError, subprocess.CalledProcessError):
        pass
    if not identity:
        identity = (
            f'stat:{stat.st_dev}:{stat.st_ino}:{stat.st_size}:'
            f'{stat.st_mtime_ns}:{stat.st_ctime_ns}'
        )
    VMLINUX_IDENTITY_CACHE.clear()
    VMLINUX_IDENTITY_CACHE[cache_key] = identity
    return identity


def gen_targets_hash(kernel_dir: str, targets: list[str], length: int = 12) -> str:
    normalized_targets = [normalize_target(kernel_dir, target) for target in targets]
    # hash by canonical func@file (without line number) to avoid collisions across same-name functions
    unique_target_func_keys = sorted(set([get_canonical_target_func_key(kernel_dir, target) for target in normalized_targets]))
    # serialize target func keys to string with sequence-sensitive
    kernel_identity = get_vmlinux_cache_identity(kernel_dir)
    content = f'{FAST_CACHE_VERSION}:{kernel_identity}:{unique_target_func_keys}'.encode('utf-8')
    # calculate SHA-256
    full_hash = hashlib.sha256(content).hexdigest()
    # truncate to specified length
    return full_hash[:length]


def stat_instru_info(info: dict):
    """Stat the instrumentation info of the kernel"""
    func_num = 0
    bb_num = 0
    funcs = set()
    for func in info:
        for loc in info[func]:
            file_path = loc.split(':')[0]
            func_hash = '%s@%s'%(func, file_path)
            funcs.add(func_hash)
            bb_num += 1
    func_num = len(funcs)
    print_t('[INFO] Detected %d funcs are instrumented with %d basic blocks'%(func_num, bb_num))


def build_allias(kernel_dir: str):
    """Build the allias of the kernel"""
    # allias.txt are generated by
    # objdump -d --no-show-raw-insn vmlinux | grep __sanitizer_cov_trace_pc | cut -d: -f1 | addr2line -afi -e vmlinux > allias.txt
    outdir = os.path.join(kernel_dir, DEF_SETS['outdir'])
    assembly_path = os.path.join(outdir, DEF_SETS['assembly'])
    allias_path = os.path.join(outdir, DEF_SETS['allias'])
    vmlinux_path = os.path.join(kernel_dir, 'vmlinux')
    assert os.path.isfile(vmlinux_path), f'vmlinux not found in {outdir}'
    if any([check_command_injection(p) for p in [vmlinux_path, assembly_path, allias_path]]):
        print_t(f'[ERROR] Command injection detected in {vmlinux_path}, {assembly_path}, {allias_path}')
        exit(1)
    cmd = f'objdump -d --no-show-raw-insn {vmlinux_path} | \
grep __sanitizer_cov_trace_pc | \
cut -d: -f1 | \
addr2line -afi -e {vmlinux_path} > {allias_path}'
    print_t(f'[INFO] Running command: {cmd}\nThis may take over ten minutes. The good news is that it only needs to be run once.')
    os.system(cmd)
    print_t(f'[INFO] {allias_path} is built')
    return allias_path


def add_instru_info(info: dict, func_name: str, loc: str, ia: str) -> None:
    """Add the instrumentation info of the target function"""
    if '?' in loc:
        # skip the unknown location
        return
    key = f'{func_name}'
    pc = hex(int(ia, 16) + 5)
    info.setdefault(key, {})
    info[key].setdefault(solve_abs_loc(loc), []).append(pc)


def build_instru_info(outdir: str, allias_path: str = None, instru_path: str = None) -> dict:
    """Build the instrumentation info of the kernel"""
    info = {}
    ia = None
    func_name = None
    if allias_path is None:
        allias_path = os.path.join(outdir, DEF_SETS['allias'])

    allias_lines = []
    print_t(f'[INFO] Loading {allias_path}...')
    with open(allias_path, 'r') as f:
        allias_lines = [line.strip() for line in f.readlines()]
    for line in allias_lines:
        if line.startswith('0x'):
            ia = line.strip()
        elif ':' in line:
            file = line.split(':')[0]
            # sometimes, there are ' (discriminator N)' after file:line_num
            line_num = line.split(':')[1].split(' ')[0]
            add_instru_info(info, func_name, f'{file}:{line_num}', ia)
        else:
            func_name = line.strip()
    # since the traversal of line is in ascending order,
    # the pc addrs in info[func_name][loc] are also in ascending order.
    if instru_path is None:
        instru_path = os.path.join(outdir, DEF_SETS['instru'])
    write_json(info, instru_path)
    print_t(f'[INFO] {instru_path} is built')
    return info


def build_pcs2funcs(info: dict, outdir: str, pcs2funcs_path: str = None) -> dict:
    """Build the pcs2funcs of the kernel"""
    if pcs2funcs_path is None:
        pcs2funcs_path = os.path.join(outdir, DEF_SETS['pcs2funcs'])
    pcs2funcs = {} # format like: {pc: set(func_hashs)}, func_hash is like: func_name@file_path

    for func in info:
        for loc in info[func]:
            func_fpath = loc.split(':')[0]
            func_hash = f'{func}@{func_fpath}'
            for pc in info[func][loc]:
                pcs2funcs.setdefault(pc, set())
                pcs2funcs[pc] |= set([func_hash])
    for pc in pcs2funcs:
        pcs2funcs[pc] = list(pcs2funcs[pc])
    write_json(pcs2funcs, pcs2funcs_path)
    print_t(f'[INFO] {pcs2funcs_path} is built')
    return pcs2funcs


def action_build(kernel_dir: str, api_mode: bool = False, rebuild: bool = False) -> dict:
    """Build the allias, instrumentation and pcs2funcs info of the kernel"""
    outdir = os.path.join(kernel_dir, DEF_SETS['outdir'])
    vmlinux_path = os.path.join(kernel_dir, 'vmlinux')
    if not os.path.isfile(vmlinux_path):
        print_t(f'[ERROR] vmlinux not found in {kernel_dir}, please make sure it contains the compiled kernel object')
        exit(1)
    instru_path = os.path.join(outdir, DEF_SETS['instru'])

    if not all([os.path.isdir(outdir), os.path.isfile(instru_path)]):
        print_t(f'[INFO] {outdir} not found or incomplete, building...')
        os.makedirs(outdir, exist_ok=True)
        build_allias(kernel_dir)
        info = build_instru_info(outdir)
        build_pcs2funcs(info, outdir)
    else:
        # command line mode
        if api_mode == False:
            print_t(f'[INFO] {outdir} already exists')
            do = input(f'[+] Rebuild it in {outdir}? (Generally there is no need to rebuild) [yes|NO]: ').lower()
            if do in ['', 'no', 'n']:
                info = read_json(instru_path)
            else:
                print_t('[INFO] Rebuilding...')
                info = build_instru_info(outdir)
                build_pcs2funcs(info, outdir)
        elif rebuild == True:
            info = build_instru_info(outdir)
            build_pcs2funcs(info, outdir)
        else:
            info = read_json(instru_path)
    return info


def solve_abs_loc(loc: str) -> str:
    """Sometimes there are relative path in file_path, like: /path/kernel/./include/xxxfile"""
    file_path, line_num = loc.rsplit(':', 1)
    return '%s:%s'%(os.path.abspath(file_path), line_num)


def split_target_location(target_loc: str) -> tuple[str, int]:
    """Split file_path:line_num with the last colon as delimiter."""
    if ':' not in target_loc:
        raise ValueError(f'invalid target location (missing line number): {target_loc}')
    file_path, line_num = target_loc.rsplit(':', 1)
    if file_path == '' or line_num == '' or not line_num.isdigit():
        raise ValueError(f'invalid target location: {target_loc}')
    line_num = int(line_num)
    if line_num <= 0:
        raise ValueError(f'invalid target line number: {target_loc}')
    return file_path, line_num


def remap_path_into_kernel_dir(kernel_dir: str, file_path: str) -> str | None:
    """
    Remap an alternate debug-info path prefix into the current kernel_dir when possible.

    Example: /root/kernels/.../case_16/fs/file.c -> /home/.../case_16/fs/file.c
    """
    norm_kernel_dir = os.path.normpath(os.path.abspath(kernel_dir))
    norm_file_path = os.path.normpath(os.path.abspath(file_path))
    if norm_file_path.startswith(norm_kernel_dir + os.sep):
        return norm_file_path

    kernel_base = os.path.basename(norm_kernel_dir)
    marker = os.sep + kernel_base + os.sep
    marker_idx = norm_file_path.find(marker)
    if marker_idx == -1:
        return None
    rel_path = norm_file_path[marker_idx + len(marker):]
    remapped_path = os.path.abspath(os.path.join(norm_kernel_dir, rel_path))
    if os.path.isfile(remapped_path):
        return remapped_path
    return None


def normalize_target_path(kernel_dir: str, file_path: str) -> str:
    """Normalize a target file path to an absolute source path under kernel_dir."""
    if os.path.isabs(file_path):
        abs_path = os.path.abspath(file_path)
    else:
        abs_path = os.path.abspath(os.path.join(kernel_dir, file_path))
    if not os.path.isfile(abs_path):
        remapped_path = remap_path_into_kernel_dir(kernel_dir, abs_path)
        if remapped_path is not None:
            return remapped_path
        raise FileNotFoundError(f'target source file not found: {file_path} -> {abs_path}')
    return abs_path


def canonicalize_kernel_source_path(kernel_dir: str, file_path: str) -> str:
    """
    Canonicalize a source path to a kernel-relative path when possible.

    This tolerates different mount prefixes for the same kernel tree, such as
    /home/.../case_N/... vs /root/.../case_N/..., by anchoring on the kernel_dir basename.
    """
    norm_kernel_dir = os.path.normpath(os.path.abspath(kernel_dir))
    norm_file_path = os.path.normpath(os.path.abspath(file_path))
    if norm_file_path.startswith(norm_kernel_dir + os.sep):
        return os.path.relpath(norm_file_path, norm_kernel_dir)

    kernel_base = os.path.basename(norm_kernel_dir)
    marker = os.sep + kernel_base + os.sep
    marker_idx = norm_file_path.find(marker)
    if marker_idx != -1:
        return norm_file_path[marker_idx + len(marker):]
    return norm_file_path


def get_canonical_target_func_key(kernel_dir: str, target: str) -> str:
    """Extract a cache-stable func_name@kernel_relative_path key from a normalized target."""
    if '@' not in target:
        raise ValueError(f'invalid normalized target (missing function name): {target}')
    func_name, target_loc = target.split('@', 1)
    file_path, _ = split_target_location(target_loc)
    canonical_path = canonicalize_kernel_source_path(kernel_dir, file_path)
    return f'{func_name}@{canonical_path}'


def get_tree_sitter_c_parser() -> Parser:
    """Get a cached Tree-sitter C parser."""
    global TREE_SITTER_C_PARSER
    if Parser is None or Language is None or tree_sitter_c_language is None:
        raise ImportError(
            'tree-sitter dependencies are required for file:line target resolution. '
            'Please install tree-sitter and tree-sitter-c.'
        )
    if TREE_SITTER_C_PARSER is None:
        parser = Parser()
        parser.language = Language(tree_sitter_c_language())
        TREE_SITTER_C_PARSER = parser
    return TREE_SITTER_C_PARSER


def get_node_text(source_code: bytes, node) -> str:
    """Get the UTF-8 text of a Tree-sitter node."""
    return source_code[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')


def extract_function_name_from_declarator(source_code: bytes, declarator_node) -> str | None:
    """Extract the function identifier from a Tree-sitter declarator chain."""
    node = declarator_node
    while node is not None:
        if node.type == 'identifier':
            func_name = get_node_text(source_code, node).strip()
            return func_name if func_name != '' else None
        node = node.child_by_field_name('declarator')
    return None


def extract_syscall_macro_function_name(node_text: str) -> str | None:
    """Extract the generated kernel function name from a SYSCALL_DEFINE* macro call."""
    node_text = ' '.join(node_text.split())
    compat_match = re.search(r'\bCOMPAT_SYSCALL_DEFINE\d+\s*\(\s*([A-Za-z_]\w*)', node_text)
    if compat_match:
        return f'__do_compat_sys_{compat_match.group(1)}'
    syscall_match = re.search(r'\bSYSCALL_DEFINE\d+\s*\(\s*([A-Za-z_]\w*)', node_text)
    if syscall_match:
        return f'__do_sys_{syscall_match.group(1)}'
    return None


def get_next_non_comment_child(children: list, start_idx: int):
    """Get the next sibling child, skipping comments."""
    for idx in range(start_idx, len(children)):
        child = children[idx]
        if child.type != 'comment':
            return child
    return None


def extract_function_name_from_call_expression(source_code: bytes, node) -> str | None:
    """Extract the callee identifier from a Tree-sitter call_expression node."""
    if node is None or node.type != 'call_expression':
        return None
    func_node = node.child_by_field_name('function')
    if func_node is None:
        for child in node.children:
            if child.type == 'identifier':
                func_node = child
                break
    if func_node is None:
        return None
    func_name = get_node_text(source_code, func_node).strip()
    return func_name if func_name != '' else None


def extract_function_name_from_recovered_header_node(source_code: bytes, children: list, child_idx: int) -> str | None:
    """Extract a function name from an error-recovered header fragment node."""
    node = children[child_idx] if 0 <= child_idx < len(children) else None
    if node is None:
        return None
    if node.type == 'call_expression':
        return extract_function_name_from_call_expression(source_code, node)
    if node.type == 'identifier':
        next_child = get_next_non_comment_child(children, child_idx + 1)
        if next_child is None or next_child.type != '(':
            return None
        func_name = get_node_text(source_code, node).strip()
        return func_name if func_name != '' else None
    return extract_function_name_from_declarator(source_code, node)


def add_error_recovered_function_ranges(node, source_code: bytes, ranges: list[dict]) -> None:
    """
    Recover function-like ranges from malformed top-level ERROR subtrees.

    Tree-sitter C sometimes degrades later kernel functions into a token stream like
    "static", "int", "func(...)", "{...}" after an earlier GNU-extension parse error.
    Recover those fragments here so file:line resolution can still pick the correct
    enclosing function by shortest range.
    """
    if node.type != 'ERROR':
        return

    children = list(node.children)
    for idx, child in enumerate(children):
        if child.type != 'compound_statement':
            continue

        saw_header_hint = False
        header_start_idx = None
        func_name = None
        func_name_line = None

        for back_idx in range(idx - 1, -1, -1):
            sibling = children[back_idx]
            if sibling.type == 'comment':
                continue
            if sibling.type in TREE_SITTER_RECOVERED_FUNC_HEADER_BARRIER_TYPES:
                break
            if child.start_point[0] - sibling.end_point[0] > 16:
                break
            if sibling.type in TREE_SITTER_RECOVERED_FUNC_HEADER_HINT_TYPES:
                saw_header_hint = True
            header_start_idx = back_idx
            if func_name is None and sibling.type in TREE_SITTER_RECOVERED_FUNC_NAME_NODE_TYPES:
                extracted_name = extract_function_name_from_recovered_header_node(source_code, children, back_idx)
                if extracted_name is not None:
                    func_name = extracted_name
                    func_name_line = sibling.start_point[0] + 1

        if not saw_header_hint or func_name is None or header_start_idx is None:
            continue
        if func_name_line is not None and func_name_line >= child.end_point[0] + 1:
            continue

        ranges.append({
            'name': func_name,
            'start_line': children[header_start_idx].start_point[0] + 1,
            'end_line': child.end_point[0] + 1,
        })


def add_nested_error_recovered_function_ranges(node, source_code: bytes, ranges: list[dict]) -> None:
    """Walk a malformed subtree and recover fragmented functions from descendant ERROR nodes."""
    if node.type == 'ERROR':
        add_error_recovered_function_ranges(node, source_code, ranges)
    for child in node.children:
        if child.type == 'ERROR' or getattr(child, 'has_error', False):
            add_nested_error_recovered_function_ranges(child, source_code, ranges)


def add_tree_sitter_function_ranges(node, source_code: bytes, ranges: list[dict]) -> None:
    """Collect top-level function ranges from a Tree-sitter subtree."""
    if node.type not in TREE_SITTER_TOPLEVEL_CONTAINER_TYPES:
        return

    if node.type == 'ERROR':
        add_error_recovered_function_ranges(node, source_code, ranges)

    children = list(node.children)
    for idx, child in enumerate(children):
        if child.type == 'function_definition':
            declarator_node = child.child_by_field_name('declarator')
            body_node = child.child_by_field_name('body')
            func_name = extract_function_name_from_declarator(source_code, declarator_node)
            if func_name is not None and body_node is not None:
                ranges.append({
                    'name': func_name,
                    'start_line': child.start_point[0] + 1,
                    'end_line': body_node.end_point[0] + 1,
                })
            if getattr(child, 'has_error', False):
                add_nested_error_recovered_function_ranges(child, source_code, ranges)
            continue

        if child.type == 'expression_statement':
            syscall_func_name = extract_syscall_macro_function_name(get_node_text(source_code, child))
            body_node = get_next_non_comment_child(children, idx + 1)
            if syscall_func_name is not None and body_node is not None and body_node.type == 'compound_statement':
                ranges.append({
                    'name': syscall_func_name,
                    'start_line': child.start_point[0] + 1,
                    'end_line': body_node.end_point[0] + 1,
                })
            continue

        if child.type in TREE_SITTER_TOPLEVEL_CONTAINER_TYPES:
            add_tree_sitter_function_ranges(child, source_code, ranges)


def get_file_function_ranges(file_path: str) -> list[dict]:
    """Parse top-level C function ranges in a source file with Tree-sitter C."""
    cache_key = os.path.abspath(file_path)
    if cache_key in SOURCE_FUNCTION_CACHE:
        return SOURCE_FUNCTION_CACHE[cache_key]

    with open(cache_key, 'rb') as f:
        source_code = f.read()

    parser = get_tree_sitter_c_parser()
    tree = parser.parse(source_code)
    ranges = []
    add_tree_sitter_function_ranges(tree.root_node, source_code, ranges)

    unique_ranges = []
    seen = set()
    for func_info in ranges:
        func_key = (func_info['name'], func_info['start_line'], func_info['end_line'])
        if func_key in seen:
            continue
        seen.add(func_key)
        unique_ranges.append(func_info)
    unique_ranges.sort(key=lambda item: (item['start_line'], item['end_line'], item['name']))

    SOURCE_FUNCTION_CACHE[cache_key] = unique_ranges
    return unique_ranges


def resolve_function_for_location(kernel_dir: str, target_loc: str) -> str:
    """Resolve file_path:line_num to func_name@abs_file_path:line_num."""
    file_path, line_num = split_target_location(target_loc)
    abs_file_path = normalize_target_path(kernel_dir, file_path)
    function_ranges = get_file_function_ranges(abs_file_path)
    matched_funcs = [
        func_info for func_info in function_ranges
        if func_info['start_line'] <= line_num <= func_info['end_line']
    ]
    if len(matched_funcs) == 0:
        raise ValueError(f'cannot resolve enclosing function for target: {target_loc}')
    if len(matched_funcs) > 1:
        matched_funcs.sort(key=lambda item: item['end_line'] - item['start_line'])
    func_name = matched_funcs[0]['name']
    return f'{func_name}@{abs_file_path}:{line_num}'


def normalize_target(kernel_dir: str, target: str) -> str:
    """
    Normalize a raw target into func_name@abs_file_path:line_num.

    Accepted input formats:
      - func_name@file_path:line_num
      - file_path:line_num
    """
    target = target.strip()
    if target == '':
        raise ValueError('target must not be empty')

    if '@' not in target:
        return resolve_function_for_location(kernel_dir, target)

    func_name, target_loc = target.split('@', 1)
    func_name = func_name.strip()
    if func_name == '':
        raise ValueError(f'invalid target (empty function name): {target}')
    file_path, line_num = split_target_location(target_loc)
    abs_file_path = normalize_target_path(kernel_dir, file_path)
    return f'{func_name}@{abs_file_path}:{line_num}'


def get_target_pc(info: dict, target: str, kernel_dir: str) -> str:
    """Get the pc address of the target"""
    func_name = target.split('@')[0]
    bb_loc = target.split('@')[1]
    abs_bb_loc = solve_abs_loc(bb_loc)

    if func_name not in info:
        return None
    locs = info[func_name]
    if abs_bb_loc in locs:
        # notice, the pc addrs in locs[abs_bb_loc] are in ascending order
        return locs[abs_bb_loc][0]
    bb_file_path, bb_line_num = split_target_location(abs_bb_loc)
    bb_rel_path = canonicalize_kernel_source_path(kernel_dir, bb_file_path)
    for loc in locs:
        loc_file_path, loc_line_num = split_target_location(loc)
        loc_rel_path = canonicalize_kernel_source_path(kernel_dir, loc_file_path)
        if loc_line_num == bb_line_num and loc_rel_path == bb_rel_path:
            return locs[loc][0]
    return None


def has_target_func_key_in_info(info: dict, target_func_key: str, kernel_dir: str) -> bool:
    """Check whether a fast-build artifact contains any instrumentation entry for target func@file."""
    func_name, target_rel_path = target_func_key.split('@', 1)
    if func_name not in info:
        return False
    for loc in info[func_name]:
        loc_file_path, _ = split_target_location(loc)
        if canonicalize_kernel_source_path(kernel_dir, loc_file_path) == target_rel_path:
            return True
    return False


def fast_artifact_covers_target_funcs(info: dict, target_func_keys: list[str], kernel_dir: str) -> bool:
    """Check whether an existing fast-build artifact covers all requested target functions."""
    return all(has_target_func_key_in_info(info, func_key, kernel_dir) for func_key in target_func_keys)


def get_missing_target_func_keys(info: dict, targets: list[str], kernel_dir: str) -> list[str]:
    """Return requested func@file keys that still have no instrumentation entries."""
    missing_func_keys = []
    for target in targets:
        target_func_key = get_canonical_target_func_key(kernel_dir, target)
        if not has_target_func_key_in_info(info, target_func_key, kernel_dir):
            missing_func_keys.append(target_func_key)
    return sorted(set(missing_func_keys))


def check_target(info: dict, target: str, kernel_dir: str, default_choice: int = None) -> str:
    """
    Check and complete a normalized target in format func_name@file_path:line_num.
    If the line number is not matched, try to find the previous closest line number.
    """
    if '@' not in target:
        raise ValueError(f'invalid normalized target (missing function name): {target}')
    func = target.split('@', 1)[0]
    bb_loc = target.split('@', 1)[1]
    fpath, lnum = split_target_location(bb_loc)
    lnum = str(lnum)
    target_rel_path = canonicalize_kernel_source_path(kernel_dir, fpath)

    if func in info:
        if bb_loc in info[func]:
            return target
        else:
            matched_instru_loc = None
            instru_sites = []
            for loc in info[func]:
                loc_file_path, _ = split_target_location(loc)
                loc_rel_path = canonicalize_kernel_source_path(kernel_dir, loc_file_path)
                if loc_rel_path == target_rel_path:
                    instru_sites.append(loc)
            assert len(instru_sites) == len(set(instru_sites))
            sorted_instru_sites = sorted(instru_sites, key=lambda s: int(s.rsplit(':', 1)[1]))
            # find the matched or previous closest location
            for instru_loc in sorted_instru_sites:
                _, instru_line = instru_loc.rsplit(':', 1)
                if int(instru_line) <= int(lnum):
                    matched_instru_loc = instru_loc
                else:
                    break
            if matched_instru_loc is None:
                raise ValueError('invalid bb_loc of target: %s'%target)
            return '%s@%s'%(func, matched_instru_loc)
    else:
        raise ValueError('invalid func of target: %s'%target)


def action_get(kernel_dir: str, targets: list[str], api_mode: bool = False, fast_mode: bool = False) -> tuple[list[str], list[str]]:
    """Get the fuzz targets and pcs"""
    resolved_targets = []
    normalize_errors = []
    for target in targets:
        try:
            resolved_targets.append(normalize_target(kernel_dir, target))
            normalize_errors.append(None)
        except Exception as e:
            resolved_targets.append(target.strip())
            normalize_errors.append(e)
    valid_resolved_targets = [
        target for target, err in zip(resolved_targets, normalize_errors)
        if err is None
    ]
    if fast_mode == False:
        info = action_build(kernel_dir, True)
    else:
        info = action_fast_build(kernel_dir, valid_resolved_targets, False) if len(valid_resolved_targets) > 0 else {}
    full_targets = []
    target_pcs = []
    # max_str_len = 0
    default_choice = 0 if api_mode == True else None
    for target, normalize_error in zip(resolved_targets, normalize_errors):
        try:
            if normalize_error is not None:
                raise normalize_error
            if fast_mode:
                target_func_key = get_canonical_target_func_key(kernel_dir, target)
                if not has_target_func_key_in_info(info, target_func_key, kernel_dir):
                    raise ValueError(
                        f'artifact incomplete: missing instrumentation entries for target function {target_func_key}'
                    )
            target = check_target(info, target, kernel_dir, default_choice)
            target_pc = get_target_pc(info, target, kernel_dir)
            if target_pc is None:
                raise ValueError(f'cannot resolve target pc for target: {target}')
        except Exception as e:
            target_pc = '0x0000000000000000'
            print_t('[ERROR] Fail to get target_pc: %s'%e)
        full_targets.append(target)
        target_pcs.append(target_pc)
        # max_str_len = max(max_str_len, len(target))
    return full_targets, target_pcs


def get_funcs_start_stop_addrs(kernel_dir: str, target_func_keys: set[str]) -> dict:
    """
    Get the start and stop addresses of target functions keyed by func_name@abs_file_path.

    We intentionally collect all symbol ranges that share the same function name.
    Source-file disambiguation is performed later by exact path matching in check_target/get_target_pc.
    This avoids relying on addr2line(symbol_start), which is unstable for many normal kernel functions.
    """
    vmlinux_path = os.path.join(kernel_dir, 'vmlinux')
    if not os.path.isfile(vmlinux_path):
        print_t(f'[ERROR] vmlinux not found in {kernel_dir}, please make sure it contains the compiled kernel object')
        exit(1)
    # Read full symbol table (-n for increasing order of addr)
    nm_cmd = f'nm -an {vmlinux_path}'
    nm_output = os.popen(nm_cmd).read()
    assert nm_output != '', f'[ERROR] fail to get symbol addr from {vmlinux_path} with nm command'

    ordered_records: list[tuple(str, str)]  = [] # list of (addr_hex, func_name)
    for line in nm_output.split('\n'):
        line = line.strip()
        if line == '':
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        addr, func = parts[0], parts[2]
        try:
            addr = hex(int(addr, 16))
        except Exception:
            continue
        ordered_records.append((addr, func))

    func_name_to_keys = {}
    d_funcs_addrs = {} # format like: {func_name@file_path: (start_addr, stop_addr)}
    for target_func_key in target_func_keys:
        func_name, file_path = target_func_key.split('@', 1)
        func_name_to_keys.setdefault(func_name, []).append(target_func_key)
        d_funcs_addrs.setdefault(target_func_key, {'start_addr': [], 'stop_addr': []})
    assert len(d_funcs_addrs) == len(target_func_keys)

    for i, (addr, func) in enumerate(ordered_records):
        # the tail records of the symbol table are usually BSS (not .text symbols), so break here
        if i == len(ordered_records)-1:
            break
        if func in func_name_to_keys:
            stop_addr = ordered_records[i+1][0]
            for target_func_key in func_name_to_keys[func]:
                d_funcs_addrs[target_func_key]['start_addr'].append(addr)
                d_funcs_addrs[target_func_key]['stop_addr'].append(stop_addr)

    for target_func_key in target_func_keys:
        assert len(d_funcs_addrs[target_func_key]['start_addr']) == len(d_funcs_addrs[target_func_key]['stop_addr'])
    fill_missing_func_addrs_from_same_source_file(kernel_dir, ordered_records, d_funcs_addrs)
    return d_funcs_addrs


def fill_missing_func_addrs_from_same_source_file(kernel_dir: str, ordered_records: list[tuple[str, str]], d_funcs_addrs: dict) -> None:
    """
    Add enclosing same-source symbol ranges for inline or non-symbol target functions.

    Inline helpers do not have their own nm symbol, but their KCOV sites are emitted
    into an outer visible function. Scanning symbols from the same source file keeps
    fast mode targeted while allowing addr2line -afi to recover the inline frame.
    """
    missing_keys = [
        func_key for func_key, addrs in d_funcs_addrs.items()
        if len(addrs['start_addr']) == 0
    ]
    if not missing_keys:
        return

    target_paths = {
        func_key: func_key.split('@', 1)[1]
        for func_key in missing_keys
    }
    source_func_names_by_path = {}
    for target_rel_path in set(target_paths.values()):
        source_path = normalize_target_path(kernel_dir, target_rel_path)
        source_func_names_by_path[target_rel_path] = {
            item['name'] for item in get_file_function_ranges(source_path)
        }

    for i, (addr, func_name) in enumerate(ordered_records[:-1]):
        stop_addr = ordered_records[i + 1][0]
        for func_key, target_rel_path in target_paths.items():
            source_func_names = source_func_names_by_path.get(target_rel_path, set())
            if func_name not in source_func_names:
                continue
            d_funcs_addrs[func_key]['start_addr'].append(addr)
            d_funcs_addrs[func_key]['stop_addr'].append(stop_addr)


def addr2line_output_contains_target_source(kernel_dir: str, target_rel_path: str, output: str) -> bool:
    """Check whether an addr2line -afi output segment contains the target source file."""
    for line in output.splitlines():
        if ':' not in line or line.startswith('0x') or line.startswith('?'):
            continue
        file_path, line_text = line.rsplit(':', 1)
        line_text = line_text.split(' ', 1)[0]
        if file_path == '' or not line_text.isdigit():
            continue
        if not os.path.isabs(file_path):
            file_path = os.path.join(kernel_dir, file_path)
        if canonicalize_kernel_source_path(kernel_dir, file_path) == target_rel_path:
            return True
    return False


def ensure_cache_dir(kernel_dir):
    """Ensure the cache directory exists"""
    cache_dir = os.path.join(kernel_dir, DEF_SETS['outdir'], 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_cached_func_allias_path(cache_dir, func_key):
    """Get the cache path for a specific normalized func_name@abs_file_path key."""
    func_name = func_key.split('@', 1)[0]
    kernel_dir = os.path.dirname(os.path.dirname(os.path.abspath(cache_dir)))
    kernel_identity = get_vmlinux_cache_identity(kernel_dir)
    cache_key = hashlib.sha256(
        f'{FAST_CACHE_VERSION}:{kernel_identity}:{func_key}'.encode('utf-8')
    ).hexdigest()[:16]
    return os.path.join(cache_dir, f'allias_{func_name}_{cache_key}.txt')


def process_func_ia_extraction_worker(task_info):
    """
    Worker function for parallel processing.
    Executes the objdump pipeline for a single function.
    """
    func_key, start_addr, stop_addr, vmlinux_path = task_info

    # Construct command pipeline
    cmd = (
        f'objdump -d --no-show-raw-insn {vmlinux_path} '
        f'--start-address={start_addr} '
        f'--stop-address={stop_addr} | '
        f'grep __sanitizer_cov_trace_pc | '
        f'cut -d: -f1 | '
        f'addr2line -afi -e {vmlinux_path}'
    )

    try:
        # Capture output instead of writing to file directly
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return func_key, result.decode('utf-8', errors='ignore')
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Worker failed for {func_key}: {e}")
        return func_key, ""


def action_fast_build(
    kernel_dir: str,
    targets: list[str],
    rebuild: bool = False,
    allow_partial: bool = False,
) -> dict:
    """Fast build the allias, instrumentation and pcs2funcs info of the target functions"""
    vmlinux_path = os.path.join(kernel_dir, 'vmlinux')
    if not os.path.isfile(vmlinux_path):
        print_t(f'[ERROR] vmlinux not found in {kernel_dir}, please make sure it contains the compiled kernel object')
        exit(1)
    outdir = os.path.join(kernel_dir, DEF_SETS['outdir'])
    os.makedirs(outdir, exist_ok=True)
    cache_dir = ensure_cache_dir(kernel_dir)
    normalized_targets = [normalize_target(kernel_dir, target) for target in targets]
    target_func_keys = sorted(set([get_canonical_target_func_key(kernel_dir, target) for target in normalized_targets]))
    cached_results = []
    missing_func_keys = []

    # Use hash to identify the targets to speedup the build process for the same targets
    targets_hash = gen_targets_hash(kernel_dir, normalized_targets)
    allias_path = os.path.join(outdir, DEF_SETS['allias']+f'_fast_{targets_hash}')
    instru_path = os.path.join(outdir, DEF_SETS['instru']+f'_fast_{targets_hash}')
    pcs2funcs_path = os.path.join(outdir, DEF_SETS['pcs2funcs']+f'_fast_{targets_hash}')

    if not rebuild and all([os.path.isfile(p) for p in [allias_path, instru_path, pcs2funcs_path]]):
        existing_info = read_json(instru_path)
        if fast_artifact_covers_target_funcs(existing_info, target_func_keys, kernel_dir):
            print_t(f'[INFO] {outdir} is up-to-date for target {normalized_targets} with hash {targets_hash}.')
            return existing_info
        print_t(
            f'[WARN] Existing fast-build artifact for hash {targets_hash} is incomplete. '
            f'Rebuilding target functions {target_func_keys}...'
        )

    print_t(
        f'[INFO] Start fast build. Checking cache for {len(target_func_keys)} functions '
        f'from targets {normalized_targets} with hash {targets_hash}...'
    )
    for func_key in target_func_keys:
        cache_path = get_cached_func_allias_path(cache_dir, func_key)
        if os.path.isfile(cache_path):
            with open(cache_path, 'r') as f:
                cached_results.append(f.read())
        else:
            missing_func_keys.append(func_key)

    if missing_func_keys:
        print_t(f'[INFO] {len(missing_func_keys)} functions missing in cache. Starting parallel build...')
        # Get the address range of the missing functions
        d_funcs_addrs = get_funcs_start_stop_addrs(kernel_dir, set(missing_func_keys))
        # Prepare the task list
        tasks = []
        for func_key in missing_func_keys:
            start_addrs = d_funcs_addrs[func_key]['start_addr']
            stop_addrs = d_funcs_addrs[func_key]['stop_addr']
            if len(start_addrs) == 0:
                print_t(f'[WARN] No symbol ranges found for {func_key} during fast build.')

            for i in range(len(start_addrs)):
                tasks.append((func_key, start_addrs[i], stop_addrs[i], vmlinux_path))

        new_func_results = {} # {func_name@file_path: [result_str_segment1, ...]}
        if len(tasks) > 0:
            num_workers = max(8, (os.cpu_count() or 16) // 2)
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_task = {executor.submit(process_func_ia_extraction_worker, t): t for t in tasks}
                total_tasks = len(tasks)

                for future in tqdm(concurrent.futures.as_completed(future_to_task), total=total_tasks, desc="Processing function instruction addrs extraction"):
                    func_key, result = future.result()
                    target_rel_path = func_key.split('@', 1)[1]
                    if addr2line_output_contains_target_source(kernel_dir, target_rel_path, result):
                        new_func_results.setdefault(func_key, []).append(result)

        print_t(f"[INFO] Updating {len(new_func_results)} functions cache...")
        for func_key, segments in new_func_results.items():
            combined_content = "".join(segments)
            with open(get_cached_func_allias_path(cache_dir, func_key), 'w') as f:
                f.write(combined_content)
            cached_results.append(combined_content)

    print_t(f'[INFO] Writing results into {allias_path}...')
    with open(allias_path, 'w') as f:
        for content in cached_results:
            f.write(content)
    print_t(f'[INFO] {allias_path} is built')

    info = build_instru_info(outdir, allias_path, instru_path)
    build_pcs2funcs(info, outdir, pcs2funcs_path)
    missing_after_fast_build = get_missing_target_func_keys(info, normalized_targets, kernel_dir)
    if missing_after_fast_build:
        if not allow_partial:
            print_t(
                f'[WARN] Fast-build artifact incomplete for target functions {missing_after_fast_build}. '
                f'Falling back to full build to resolve inline or non-symbol targets.'
            )
            return action_build(kernel_dir, True)
        print_t(
            f'[WARN] Fast-build artifact incomplete for target functions {missing_after_fast_build}. '
            'Returning the partial artifact so unavailable intermediate waypoints can be removed.'
        )
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build the instrumentation info and get the fuzz targets.')
    subparsers = parser.add_subparsers(dest='action')

    # build action
    build_parser = subparsers.add_parser('build', help='build the instrumentation info')
    build_parser.add_argument('-k', '--kernel_dir', type=str, required=True, help='directory of kernel object')
    build_parser.add_argument('-a', '--api_mode', action='store_true', help='test api mode')
    build_parser.add_argument('-r', '--rebuild', action='store_true', help='rebuild the instrumentation info')

    # get action
    get_parser = subparsers.add_parser('get', help='get the fuzz targets')
    get_parser.add_argument('-k', '--kernel_dir', type=str, required=True, help='directory of kernel object')
    get_parser.add_argument(
        '-t', '--target', type=str, nargs='+', required=True,
        help='Targets in format funcname@filepath:line or filepath:line; multiple targets can be specified with blank separation'
    )
    get_parser.add_argument('-a', '--api_mode', action='store_true', help='test api mode')
    get_parser.add_argument('-f', '--fast_mode', action='store_true', help='fast mode (no need to do build action, very fast)')
    get_parser.add_argument('-c', '--comprehensive_mode', action='store_true', help='comprehensive mode (TODO)')
    args = parser.parse_args()

    if args.action == 'build':
        info = action_build(args.kernel_dir, args.api_mode, args.rebuild)
        stat_instru_info(info)
        print_t('[Success] analysis results are generated at %s'%DEF_SETS['outdir'])
    elif args.action == 'get':
        full_targets, target_pcs = action_get(args.kernel_dir, args.target, args.api_mode, args.fast_mode)
        print('\n| %0-60s | %0-18s |'%('func_name@file_path:line_num', 'pc_addr'))
        for i in range(len(target_pcs)):
            print('| %0-60s | %0-18s |'%(full_targets[i], target_pcs[i]))
        print(f"\n[For SyzPilot-fuzzer:]")
        s = ",".join([f"\"0x{pc[-8:]}\"" for pc in target_pcs])
        print(f"[{s}]")
        print(f"\n[For bench_parser:]")
        s = " ".join([f"reachability-0x{pc[-8:]}" for pc in target_pcs])
        print(s)
    print('\n')
    print_t('[√] Done')
