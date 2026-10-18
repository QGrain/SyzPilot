# SyzPilot benchmark assets

`configs/` mirrors the locally available, non-PoC case assets from
`SyzPilot-experiments/configs`: kernel `.config`, pinned `.commit`, bug
`.title` and `.report`, and `.hash` where available. The source currently
has cases 1–43 and 46–70; cases 44–45 are absent. The source also lacks
`case_62.report` and `.hash`, and `case_67`–`case_70.hash`. No placeholder
files have been synthesized. Case 62 can be compiled but cannot follow the
report-derived waypoint workflow without its missing report.

The benchmark does not contain target PoCs. Do not use a target reproducer
as a fuzzing seed or guidance input in functional or comparative runs.
`compile_targets.csv` is the full available-case compilation list; for a
smaller functional run, start with [`mini-benchmark/`](../mini-benchmark/).

## Published report-derived waypoints

`waypoints/` contains the ordered targets and target PCs extracted from every
case that has a compiled `vmlinux`, title, and report. The files are generated
from the report and the corresponding local kernel binary; they do not use a
target PoC. Case 62 is listed as skipped because its report is unavailable.

Regenerate the complete set with:

```bash
conda activate syzpilot
python analyzer/extract_benchmark_waypoints.py \
  --benchmark-dir benchmark \
  --kernel-root /path/to/compiled/cases
```

Report frames retain their function-relative instruction offsets. When one
report includes an offset, the resolver selects the closest preceding KCOV PC
after mapping the report's relative function position to the locally rebuilt
outer symbol. Exact symbols take priority, and a compiler-normalized clone is
accepted only when it is unique. This is a best-effort cross-build heuristic. Reported inline
function and source-file context is corroborating evidence rather than a hard
filter because the KCOV call can precede an inline call in the same basic
block. A graph-derived or manually specified `function@file:line` target has
no dynamic report offset; it falls back deterministically to the
lowest-address candidate and reports the ambiguity at debug level. Address
order must not be interpreted as control-flow depth.

Every waypoint JSON entry records the applied `resolution_method`. A fully
inlined frame without an independent local ELF symbol uses an explicitly
marked source-line fallback. When the crash frame itself has no KCOV point,
`report_target`, `final_target`, and `final_target_kind` identify the nearest
instrumented primary-crash caller used as a proxy. This applies to case 20
(`remove_wait_queue` -> `ep_remove_wait_queue`) and case 28 (`__free_pages` ->
`watch_queue_set_size`) in the current benchmark build.
