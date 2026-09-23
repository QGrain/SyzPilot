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
