# KallGraph integration and optional waypoint tools

SyzPilot's functional path requires **only the upstream KallGraph** executable to
produce `callgraph.csv`. `brain/static_analyzer.py` reads that CSV; it does not
invoke either tool in `KallGraph.diff`. Applying this patch is therefore
**optional**, not a prerequisite for building or running the core artifact.
The patch preserves two research utilities for graph-derived waypoint analysis
without changing upstream KallGraph's build or output format.

## Reproduce the KallGraph base

The patch is relative to QGrain/KallGraph commit
`0d8463a5a08d8ea4c74552e7b6b656302846d3f4`:

```bash
git clone https://github.com/QGrain/KallGraph.git KallGraph
cd KallGraph
git checkout 0d8463a5a08d8ea4c74552e7b6b656302846d3f4
git apply --check /path/to/SyzPilot/analyzer/KallGraph.diff
git apply /path/to/SyzPilot/analyzer/KallGraph.diff
```

Use a clean checkout. A mismatched commit or previously applied patch should
make `git apply --check` fail; do not force it with `--reject` or `--3way`.
The patch adds only `analyze_waypoints.py` and `src/DominatorAnalyzer.cpp`.
It intentionally excludes local absolute LLVM/SVF paths, local `.gitignore`
changes, evaluation scripts, reports, and the experimental V2 analyzer.

## Build and generate the required call graph

Follow the checked-out KallGraph `README.md` to install LLVM 14.0.6, its
bundled patched SVF 2.5, and Z3. Set your **own** LLVM/SVF paths as required by
that upstream build; the optional SyzPilot patch makes no site-specific build
configuration changes. Compile KallGraph and run it on a kernel `bc.list`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DSVF_DIR=/path/to/KallGraph/SVF-2.5
cmake --build build --target KallGraph -j 4
build/bin/KallGraph @/path/to/bc.list -OutputDir=/path/to/output -ThreadNum=4
```

KallGraph creates a timestamped subdirectory under `-OutputDir`; pass that
subdirectory's `callgraph.csv` to SyzPilot's static analyzer. The Python
bridge and report-derived fallback are documented in the main artifact.

For a source installation outside Docker, install
`requirements/requirements-brain.txt` before using `get_targets.py`.
Its `function@file:line` resolver requires the pinned `tree-sitter` and
`tree-sitter-c` packages from that requirements file.

## Optional graph-derived waypoint analysis

For the Python utility, install `python-igraph` in the interpreter used to run
the script. It supports target-only `backbone` mode and writes a JSON result:

```bash
python -m pip install python-igraph
python analyze_waypoints.py \
  --callgraph /path/to/output/<timestamp>/callgraph.csv \
  --target 'function@relative/source.c:line' \
  --mode backbone \
  --output /path/to/waypoints.json
```

The optional C++ utility depends on the igraph C development library and can
be built independently. It is **not** added as an unconditional CMake target,
so reviewers who only need the functional path need not install igraph:

```bash
g++ -std=c++17 -O2 src/DominatorAnalyzer.cpp \
  -o DominatorAnalyzer $(pkg-config --cflags --libs igraph)
./DominatorAnalyzer --csv /path/to/callgraph.csv \
  --entry syscall_entry --target target_function --stats
```

These tools analyze a static call graph only. They do not consume target PoCs,
train the model, or supply runtime guidance automatically.

Graph-derived nodes normally identify a function or source callsite, not the
dynamic instruction offset present in a crash report. If a graph-derived or
manual `function@file:line` target maps to multiple KCOV PCs,
`get_targets.py` uses the lowest-address candidate as a deterministic fallback
and prints the ambiguity with `--debug`. The lowest address is not guaranteed
to be the earliest, shallowest, or mandatory control-flow block.

Report-derived extraction is more precise because
`waypoints_extractor.py` retains each frame's `+offset/size` and inline stack.
It maps the reported relative function position to the locally rebuilt symbol,
then selects the closest preceding KCOV PC from that outer function.
An exact local ELF symbol is preferred; a compiler-normalized clone is used
only when it is unique, while ambiguous sibling clones fail publication.
Normalizing by both sizes is a best-effort tie-breaker when compiler output
changes; it does not assume that two different builds have
instruction-for-instruction equivalent layouts. Inline functions and source
files are retained as corroborating context rather than hard filters: a KCOV
call is at the basic-block entry, whose DWARF inline stack can precede an
inline call later in the same block.

If a report frame has no independent local ELF symbol because it was fully
inlined, function-relative offset mapping is unavailable. The publisher keeps
the frame only when local DWARF exposes its source-line KCOV PCs, selects the
lowest-address source-line PC, and records
`inline-source-line-lowest-address` in that waypoint. If the report's crash
frame has no KCOV point at all, the extractor uses the nearest instrumented
caller on the primary crash trace as an explicit proxy. Published JSON records
both `report_target` and `final_target_kind`; the proxy is never presented as
the original crash frame.

The published value is the runtime KCOV PC (the return address immediately
after the compiler-inserted coverage call), not the address of that call
instruction. Consequently, running `addr2line` on the published PC can display
the following source line; the extractor resolves the source location at the
coverage call site before converting it to KCOV's runtime PC convention.
Use `extract_benchmark_waypoints.py` to regenerate portable JSON for a set of
compiled benchmark kernels:

```bash
python analyzer/extract_benchmark_waypoints.py \
  --benchmark-dir benchmark \
  --kernel-root /path/to/compiled/cases
```

Each generated waypoint records its `resolution_method`. Offset-bearing frames
with a usable local symbol range fail publication if relative-position
resolution fails; they do not silently degrade to a source-line candidate.

## Patch validation

The patch was checked against a clean checkout of the exact base commit with
`git apply --check` and `git apply`. With local LLVM 14 and built SVF libraries,
the patched checkout configured with CMake and built the unmodified KallGraph
target. The C++ utility was compiled separately with igraph 0.9.6, and the
Python utility passed a syntax parse. Python runtime execution was not checked
because `python-igraph` is not installed in the validation environment. These
checks do not establish end-to-end waypoint quality; that remains a separate
evaluation question.
