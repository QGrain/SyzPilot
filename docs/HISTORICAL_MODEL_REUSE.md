# Historical models for Functional diagnostics

Historical target-specific classifiers can avoid a new training GPU during a
Functional exercise, but they are not interchangeable with a cold online
training run. Keep replay clearly labeled and separate from every fair
directed-fuzzing effectiveness comparison. A target PoC must never be used as
training or fuzzing input.

## Current evidence for the mini-benchmark

The deployed trainer and serving contract currently support Stages 1 and 2.
Stage 1 predicts reached versus unreachable; it does not distinguish deeper
waypoints. No validated Stage-3 model or three-stage serving timeline exists.

| Case | Retained model evidence | Reuse status |
| --- | --- | --- |
| 21 | Accuracy 0.9931, macro F1 0.9845; validation counts 170 unreachable / 1,130 reached | Provisional: an older manifest lacks today's signature-disjoint promotion record. |
| 25 | Postfix-guided round 14: accuracy 0.8588, macro F1 0.8587; counts 201 / 146 | Provisional: historical promotion evidence is incomplete. |
| 36 | Accepted Stage 1 and sparse Stage 2 now exist from one chronological run. Stage 2 active-class macro F1 is 0.7247 and final-target recall is 0.7677. | Best reuse candidate, but only for an explicitly labeled case-36 diagnostic; intermediate waypoint recalls remain inadequate. |

Earlier, all 31 retained case-36 Stage-2 promotion decisions were rejected,
chiefly for low macro F1 and reached-class recall; 30 also lacked sufficient
class support. The 2026-09-26 run produced the first accepted sparse Stage-2
candidate after Stage 1 in the same chronology. Its active classes were
unreachable, shallow-reached, and final-target-reached (`[0, 1, 4]`). Classes
2 and 3 had only 29 and 14 validation examples and recalls of 0.069 and 0.0,
so this result validates sparse-stage promotion rather than a fully learned
waypoint classifier. Do not choose a model merely because its checkpoint
exists, or switch stages according to wall-clock time without an accepted
chronological model from the same training history.

The accepted case-36 Stage-1 checkpoint and TorchScript artifact are about
497 MB (474 MiB) each. They are preserved in a local model store outside Git, alongside
their manifest, promotion decision, and historical manager config. The
checkpoint SHA-256 is
`13c4c6ab0313d753168109501ccf95f6fbecbb82d4a92a426d1f439a00d758b1`;
the TorchScript SHA-256 is
`43643a6b1f2ba44fac7ceda41c54dfa9cb15ab9f925d9922409452d674095724`.
The preservation copy is not a public release or an enabled runtime mode.

The new accepted case-36 Stage-2 checkpoint and TorchScript artifact are also
about 497 MB each and remain outside Git with the right-censored run evidence.
Their SHA-256 values are
`b2a04bdc4bc92582280c0ac5520b3794f8e8fbb02924fa3291815e5816d6e80b`
and
`dbc0f6c518ac96a05968ec26e679f9d7f907d60d2366eebc82fa2df3183dd99e`.
Stage 1 trained for 617 seconds and Stage 2 for 769 seconds; both reserved at
most about 18.1 GB of CUDA memory. The online lifecycle intentionally removed
the superseded Stage-1 binaries after committing v2, while retaining its
manifest and promotion decision. Therefore this run alone cannot replay the
exact v1-to-v2 serving timeline. The previously preserved Stage-1 diagnostic
remains a separate historical origin and must not be presented as the v1 from
this chronology.

## Mandatory replay gates

Before implementing or enabling replay, require:

1. A compatible kernel commit and build, identical ordered target-PC chain,
   target function, curriculum schema, class count, label order, tokenizer
   fingerprint, and 1,024-token inference contract.
2. Verified artifact hashes, an accepted promotion record with class support
   and per-class recall, and TorchScript/checkpoint prediction parity on
   held-out programs. Older models without these records remain provisional.
3. A separately configured inference-only task that does not quietly mark
   online training as complete. Report its historical training corpus and
   origin, and keep its metrics separate from a cold online-learning arm.

A three-stage time-course replay is not currently possible. The current
implementation and retained chronology contain only Stages 1 and 2, and the
accepted Stage-2 model does not learn the two sparse intermediate classes well.
Build time-course replay only after every intended snapshot from one chronology
is deliberately retained with activation times, model versions, compatibility
metadata, and validation provenance. Until then, case-36 historical replay is
only an explicitly labeled Functional warm-start or inference-only diagnostic.
