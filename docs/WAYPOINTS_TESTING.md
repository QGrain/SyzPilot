# Waypoint Extractor Regression Standard

## Purpose

Every full benchmark run must produce one published XLSX workbook plus canonical
JSON/JSONL sources. JSON is the machine-readable source of truth; the workbook
is the single review and paper-analysis view. Console logs and the internal
static workbook are not primary results.

The standard separates two questions:

1. **Correctness**: Is the target, report section, source location, PC mapping,
   and ordering valid?
2. **Effectiveness**: Is the valid chain useful for staged reachability guidance?

No unique optimal intermediate chain exists for every report. Effectiveness
metrics must not override correctness failures.

## Running a Full Regression

Use the orchestrator and preserve the generated run directory:

```bash
conda activate syzpilot
RUN_ID=20260829_v3_full
python analyzer/run_waypoint_benchmark.py \
    --run-dir agent_analysis/waypoint_runs/$RUN_ID \
    --codex-home /path/to/codex-home \
    --model gpt-5.6-sol \
    --effort xhigh \
    --max-concurrency 6 \
    --semantic-weight 0.30 \
    --hit-weight 0.20 \
    --length-weight 0.50 \
    2>&1 | tee agent_analysis/waypoint_runs/$RUN_ID.log
```

The default output is:

```text
agent_analysis/waypoint_runs/<UTC_TIMESTAMP>_<GIT_SHA>/waypoints_evaluation.xlsx
```

`--codex-home` selects the Codex authentication/configuration home used by every
SDK thread in that run. Use a dedicated, access-controlled directory and do not
launch concurrent shards that write the same JSONL output.

The run also contains canonical static, agentic, blind-score, and evaluation
JSON/JSONL artifacts, logs, a compact manifest, and a generated `README.md` for
paper-revision agents. The static XLSX remains under `.cache/` only to render
the final combined workbook.

Agentic stages use a separate environment because the Codex SDK currently
requires a newer Pydantic than the Brain environment:

```bash
conda create -n syzpilot-agent python=3.11
conda activate syzpilot-agent
pip install -r requirements/requirements-agent.txt
```

Install the system `bubblewrap` package before agentic runs. The wrapper creates
an empty home namespace, exposes only the selected kernel cases and configs, and
copies authentication into an ephemeral `CODEX_HOME`. Every exposed kernel case
overlays `SyzPilot-analysis/` with an empty temporary filesystem so script-derived
resolver caches remain hidden. Project files, Codex history, other home data, and
KCOV results are not visible to the blind agent.

A targeted development run may use `--case-ids 1,4,41`, but it is not a full
regression and must not replace the all-case artifact.

Each stage is resumable. Static reuse validates the title, report, referenced
source files, and kernel ELF build identity without rereading multi-gigabyte
artifacts. Agent-module `--resume` is deliberately explicit. The extractor reuses
a successful record only when model, effort, benchmark metadata, and current input
presence match. The scorer additionally requires the candidate method map and full
candidate-chain snapshot to match; failed records are retried. The orchestrator
directly reuses a completed agent step only when its command, lightweight input
file states, and output state still match, and passes `--resume` only for a matching
interrupted or retrying step. Start a new run directory, or omit `--resume` when
invoking an agent module directly, after changing untracked source evidence. The
SDK uses one ephemeral thread per case with web disabled, `read_only`, and
`deny_all`.

Successful agentic extraction and blind-score records use schema 2.2. After the blind agent
returns a target-first chain, a trusted host-side resolver verifies that index
zero maps to a nonzero PC. An unresolvable target is returned only to the same
thread for a bounded, evidence-preserving repair. Resolver subprocesses have a
bounded timeout and process-group cleanup. This gate proves instrumentation
observability, not semantic equivalence to the configured Bug Position. The
independent blind target-fidelity judgment supplies that semantic evidence; an
agentic target receives the dynamic target-hit bonus only when the judgment is
`exact` or `resolvable_proxy`.

## Workbook Schema

The published workbook is intentionally compact. Machine-readable JSON/JSONL is
the source of truth for complete provenance; the generated
`README.md` explains the sheets intended for paper revision.

### `Run_Metadata`

Records the run ID, UTC timestamp, git state, command, input roots, chain
direction, model configuration, and extractor parameters. Internal cache fields
are not exported to the workbook.

### `Cases`

Contains one row for every selected benchmark row, including missing reports and
failed extractions. It records:

- benchmark identity, title, commit, and Bug Position;
- status, failed stage, and exception;
- the chain, aligned PC list, and count after every paper phase;
- the implementation-only BB-resolution stage;
- the retained engineering `remove_outliers` stage;
- the final chain and terminal metadata;
- both exact-location and enclosing-function relations to Bug Position.

All displayed chains use **target-to-entry** order. Every `Listed PCs` cell uses
the same order and aligns item by item with its chain. `Fuzzer target_pcs` is
stored separately in the actual **entry-to-target** order used for deepest-reached
labels.

### `Waypoints_Long`

Contains one row per case, stage, and waypoint. It records source and resolved
locations, 64-bit and transmitted 32-bit PCs, linked-list indices, inline state,
BB metrics, value score, and whether the node survives the next stage. Script
stages leave `Causal Phase` blank. Agentic extraction records phase metadata in
a sidecar while retaining the same ordered chain and PC interface.

### `Validation`

Contains case-level invariant results with `error` or `warning` severity. Errors
identify invalid or unusable artifacts. Warnings identify semantics that require
review but may have an accepted explanation.

### `Summary`

`Summary` aggregates coverage, failures, Bug Position relations, stage lengths,
and failed validations. `Changes`, `Manual_Audit`, and `Legend` are omitted from
the published workbook; baseline and oracle data remain available in canonical
inputs when needed.

### Quality Sheets

`Agentic_Extraction` and `Agentic_Waypoints` preserve the independent agentic
candidate, report evidence, causal phases, resolved PCs, and normalization drops.
`Evaluation` contains one row for every script rule stage and one agentic-final
row per successful case. `Evaluation_Waypoints` expands each candidate into
aligned nodes and KCOV hit outcomes. `Rule_Level_Summary` reports stage type,
cross-case change prevalence, candidate and operational-label length, compression,
target retention, KCOV hit statistics, and one combined quality score. The paper
table uses only rows with `Extraction Method=script`; `agentic:final` remains an
artifact-only optional comparison.

Candidate and operational compression use a paired micro-average against the raw
call trace: `1 - sum(stage lengths) / sum(raw lengths)` over the same case set.
`Coverage_Runs` retains the compact KCOV statuses, counts, timings, and paths
needed to interpret dynamic scores.

Agentic extraction is an evidence-grounded candidate for reports whose serial
stack is insufficient. It is not a unique ground truth. Any allocation, free,
handoff, or trigger proxy must say what it proves and what it does not prove.
Every evaluated chain must have nonzero, globally unique KCOV PCs. An agentic
chain must place its unique `configured_target` phase first; lifecycle evidence
then follows in reverse causal order so reversing the array yields the same
entry-to-target curriculum consumed by deepest-reached labeling.

The blind agent returns categorical target fidelity, section fidelity, causal
coherence, and parsimony judgments. The deterministic mapper assigns these up to
85 points and adds up to 15 PC-observability points to form semantic score `S`.
The agent never receives KCOV evidence or method identity.

For `h` KCOV-hit waypoints among `r` nonzero unique operational PCs, hit-quality
score `H` is:

```text
H = 50 + 30*h/r + 10*min(h/4, 1) + 10*target_hit
```

The 50-point floor treats a single PoC miss as inconclusive, while hit ratio,
absolute hit count up to `K_min=4`, and configured-target coverage add positive
evidence. For the unnormalized rule-stage candidate count `n`, effective-length
score `L` directly rewards the rules that shorten the candidate chain:

```text
L = 100*min(4/n, 1)
```

The paper-facing baseline uses the following linear formula:

```text
Quality = 0.30*S + 0.20*H + 0.50*L
```

The length component has the largest weight and saturates at the existing
`K_min=4`. Zero-PC and duplicate-PC normalization is not applied to this component
because it is an output normalization rather than a paper rule; it remains active
for KCOV comparison and the actual fuzzer label array. Quality has no additional
cap and is unavailable without comparable KCOV evidence. `Overall Waypoint Hit
Rate` is `sum(h)/sum(r)` over coverage-compatible cases, so every operational PC
has equal weight. The older dynamic coverage score remains in `Evaluation` only
as a diagnostic trajectory.

The main comparison uses only the union of per-call coverage because that is the
coverage merged by the fuzzer's reachability labeler. `extra` coverage is kept as
separate supporting evidence. `get_targets.py` returns the KCOV return PC, while
`syz-execprog -coverfile` records `PreviousInstructionPC`; on amd64 the evaluator
therefore compares each target against `target_pc - 5`.

Only runs with compatible architecture and coverage-PC semantics, disabled KASLR,
and `COMPLETE`, `PARTIAL`, or valid completed `EMPTY` coverage contribute to the score. A crash
before `syz-execprog` writes any coverfile is `UNAVAILABLE`, not a zero-coverage
observation. This distinction prevents collection failure from being scored as
negative reachability evidence.

Existing KCOV unions are reused when waypoint chains or scoring logic changes.
Kernel binaries are treated as fixed experiment inputs and are not fingerprinted
by the coverage evaluator. To recollect coverage after changing a PoC, kernel,
or collector, use a new output directory or pass `--overwrite` to the coverage
collector. Raw/union coverage is retained for hit-source evidence. An exact
but non-reproducing PoC run remains useful lower-bound coverage evidence; it is
not described as vulnerability ground truth.

To refresh reporting metrics in an existing canonical run without rerunning
extraction, Codex agents, target resolution, or QEMU, use:

```bash
python -m analyzer.waypoint_evaluation refresh-reporting \
  --json agent_analysis/waypoint_runs/<RUN>/waypoints_evaluation.json \
  --workbook agent_analysis/waypoint_runs/<RUN>/waypoints_evaluation.xlsx
```

The command validates the Run ID, case set, and candidate set, then replaces only
`Evaluation` and `Rule_Level_Summary`, updates reporting metadata and README, and
preserves all other workbook sheets and columns.

For a non-destructive weight sensitivity variant, provide three nonnegative
weights that sum to 1 and an output directory:

```bash
python -m analyzer.waypoint_evaluation refresh-reporting \
  --json agent_analysis/waypoint_runs/<RUN>/waypoints_evaluation.json \
  --workbook agent_analysis/waypoint_runs/<RUN>/waypoints_evaluation.xlsx \
  --semantic-weight 0.25 --hit-weight 0.25 --length-weight 0.50 \
  --output-dir agent_analysis/waypoint_runs/<RUN>_quality_sensitivity
```

The selected weights are recorded in evaluation schema 2.4 JSON, workbook
metadata, the generated README, and the full-run manifest. Omitting the flags
uses the paper baseline `0.30/0.20/0.50`.

## Correctness Gates

The following properties are regression gates:

- every benchmark row has a result row, even when an input is missing;
- filtering phases are ordered subsequences of their predecessors;
- displayed chains and aligned PC arrays have equal lengths;
- final PCs resolve to nonzero KCOV instrumentation sites;
- final transmitted 32-bit PCs are globally unique within a chain;
- final length does not exceed `K_max`;
- structured frames use valid `func@file:line` locations;
- terminal and fallback resolution are explicit rather than silently inferred;
- serial stacks preserve report order, while any future composite trace must
  preserve order within each causal phase;
- agentic chains mark the configured target exactly once at target-to-entry
  index zero, and the declared proxy resolves to a verified nonzero PC;
- report section boundaries exclude headings, register dumps, page-owner data,
  and unrelated metadata.

Falling below `K_min`, a Bug Position function away from the terminal, or a title
function mismatch is a warning until the authoritative target oracle has been
audited. A configured terminal must never be removed merely because it is a
generic kernel function.

## Ground Truth Strategy

A single hand-written chain is too restrictive because several intermediate
call-sites can represent equivalent progress. Use a layered oracle instead:

1. **Target oracle for every report case**: confirm Bug Position, the valid
   call-site basic block, and any inline or uninstrumented caller fallback.
2. **Section oracle for every multi-section report**: mark crash, allocation,
   free, handoff, related-work, and forbidden report boundaries without requiring
   a unique final chain.
3. **Stratified chain audit set**: record required, allowed, and forbidden nodes
   for serial, UAF, invalid-free, inline-target, worker/softirq, and generic-sink
   examples. Keep a frozen holdout subset.
4. **Dynamic observability oracle**: use reproducer/KCOV evidence to verify that
   the terminal and important phase proxies are observable. This does not prove
   that the chain is optimal.

Every regression requires review of all target or PC changes, all invariant
failures, all changed multi-section chains, and a stratified sample of unchanged
serial chains.

Accepted audit decisions must be promoted from the workbook into
`benchmark/waypoints_oracle.json`; editing only one generated XLSX does not create
a reusable oracle. The oracle begins empty intentionally and must not be filled
with generated output as if that output were ground truth.

## Effectiveness Evidence

The following are proxy metrics, not correctness oracles:

- stage retention and final length distributions;
- generic utility, inline frame, same-source concentration, and PC duplication;
- waypoint hit prevalence and phase observability under KCOV;
- training class occupancy, calibration, and deepest-label distribution;
- target TTH, successful hit runs, hit count, and bug reproduction.

An optimization is accepted only when correctness gates remain satisfied and its
claimed benefit is supported by the appropriate proxy or fuzzing evidence.
