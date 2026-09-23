# Optional Brain GPU admission

`brain/gpu_admission.py` is an opt-in startup helper, not a running GPU
scheduler. It samples all `nvidia-smi`-visible devices three times, uses the
worst free-memory and utilization observations, and selects two distinct
cards: the least busy eligible trainer and the least busy remaining
TorchServe/attribution card. It never starts a trainer or imports PyTorch.

The initial budgets are 24,000 MiB free / at most 60% utilization for
training and 12,000 MiB free / at most 85% for serving plus attribution.
Historical training manifests show about 16.86 GiB peak CUDA reservation
with the current 64x2 batch profile. One steady TorchServe worker used about
1,081 MiB. TorchServe plus IG peaked at 5,349 MiB in a single-client profile
and 5,481 MiB with four concurrent inference clients; the latter profile had
no request failures. The 12,000 MiB serving floor therefore retains more than
2x measured headroom. Tune these limits only with the relevant workload
profile and retained telemetry.

From the repository root, first inspect a dry run:

```bash
env -u CUDA_VISIBLE_DEVICES \
  -u SYZPILOT_TRAINING_GPU_IDS \
  -u SYZPILOT_TRAINING_FALLBACK_GPU_IDS \
  -u SYZPILOT_INFERENCE_GPU_ID \
  -u SYZPILOT_ATTRIBUTION_GPU_ID \
  python brain/gpu_admission.py
```

When ports, task state, and the selected cards have been checked, launch the
Controller through the same helper:

```bash
env -u CUDA_VISIBLE_DEVICES \
  -u SYZPILOT_TRAINING_GPU_IDS \
  -u SYZPILOT_TRAINING_FALLBACK_GPU_IDS \
  -u SYZPILOT_INFERENCE_GPU_ID \
  -u SYZPILOT_ATTRIBUTION_GPU_ID \
  python brain/gpu_admission.py -- python brain/controller.py --port 48000
```

Use `--training-max-utilization`, `--serving-max-utilization`, and the
corresponding `--*-min-free-mib` options to set measured budgets. Existing
`SYZPILOT_TRAINING_*` threshold environment variables are inherited unless
the CLI overrides them. The helper prints the assignment and exits without
launching if it cannot verify two eligible GPUs, CUDA/NVML PCI identities,
or that the Controller/TorchServe ports and TorchServe global PID file are
unoccupied. An existing `config.properties` must describe those same five
TorchServe ports; the helper will not assume that a stale config is safe.
MIG configurations are currently rejected because the existing
Controller expects physical device indices. The helper does not remove a
stale PID file; inspect it manually before retrying.

The Controller independently enforces the same physical-index identity
contract even when launched directly: it rejects `CUDA_VISIBLE_DEVICES`, sets
an absent `CUDA_DEVICE_ORDER` to `PCI_BUS_ID` (and rejects any other explicit
value), then compares every CUDA ordinal with the corresponding `nvidia-smi`
PCI bus ID before TorchServe starts. This validation checks identity, not
current capacity; use the optional launcher when startup load admission is
required.

The Controller still rechecks the selected training GPU at every training
request and defers a busy run through the Receiver's durable retry path.
TorchServe stays on the selected serving GPU for the Controller lifetime;
there is no live migration. Startup checks cannot reserve GPUs. The
Controller now launches TorchServe in the foreground with a private PID-file
directory and a validated private configuration snapshot. Shutdown waits for
every live member of its owned process group, escalating to SIGKILL only while
ownership is still verifiable; an unconfirmed cleanup retains ownership and
raises an error. Ownership is independent of the successful-start flag, so a
partially started service is retried during final cleanup. Controller shutdown
attempts all owned resource classes before reporting aggregated failures, and
the health endpoint requires both the owned Java process with all five
listeners and a valid 0.5-second-timeout management response. Readiness
additionally
requires its Java process to own all five configured listening ports; an
inherited `TS_CONFIG_FILE` cannot override the checked configuration. This
prevents a competing server's `/models` response from being accepted as our
own. Direct Controller launches use the same ownership checks but retain
their fixed GPU configuration and 20% training-utilization default.

Historical-model inference-only mode is separate and is not implemented by
this launcher. A pre-trained target-specific model must never silently
replace the online-trained arm of a fair directed-fuzzing comparison.
