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

## Patch validation

The patch was checked against a clean checkout of the exact base commit with
`git apply --check` and `git apply`. With local LLVM 14 and built SVF libraries,
the patched checkout configured with CMake and built the unmodified KallGraph
target. The C++ utility was compiled separately with igraph 0.9.6, and the
Python utility passed a syntax parse. Python runtime execution was not checked
because `python-igraph` is not installed in the validation environment. These
checks do not establish end-to-end waypoint quality; that remains a separate
evaluation question.
