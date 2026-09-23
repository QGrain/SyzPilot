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

| Case | Retained Stage-1 evidence | Reuse status |
| --- | --- | --- |
| 21 | Accuracy 0.9931, macro F1 0.9845; validation counts 170 unreachable / 1,130 reached | Provisional: an older manifest lacks today's signature-disjoint promotion record. |
| 25 | Postfix-guided round 14: accuracy 0.8588, macro F1 0.8587; counts 201 / 146 | Provisional: historical promotion evidence is incomplete. |
| 36 | Accuracy 0.9279, macro F1 0.9241; counts 2,357 / 3,373; class recalls 0.8553 / 0.9787 | First preservation candidate: promotion accepted on 5,730 signature-disjoint validation examples. |

All 31 retained case-36 Stage-2 promotion decisions were rejected, chiefly
for low macro F1 and reached-class recall; 30 also lacked sufficient class
support. Do not choose a model merely because its checkpoint exists, or
switch stages according to wall-clock time without an accepted chronological
model from the same training history.

The accepted case-36 Stage-1 checkpoint and TorchScript artifact are about
497 MB (474 MiB) each. They are preserved in a local model store outside Git, alongside
their manifest, promotion decision, and historical manager config. The
checkpoint SHA-256 is
`13c4c6ab0313d753168109501ccf95f6fbecbb82d4a92a426d1f439a00d758b1`;
the TorchScript SHA-256 is
`43643a6b1f2ba44fac7ceda41c54dfa9cb15ab9f925d9922409452d674095724`.
The preservation copy is not a public release or an enabled runtime mode.

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

A three-stage time-course replay is not currently possible. Build it only
after genuine Stage-2/3 snapshots are promoted from a single chronological
run and their activation times, model versions, and validation provenance are
retained. Until then, case-36 Stage-1 replay is only a candidate for an
explicitly labeled Functional warm-start diagnostic.
