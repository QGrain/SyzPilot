# SyzPilot functional mini-benchmark

This source-only subset provides three report-grounded Linux targets for
functional checks. It contains kernel configs, reports, titles, commit IDs,
and hashes, but no target PoCs, compiled kernels, guest disks, or fixed PC
addresses. Never use a target PoC or reproducer as fuzzing input or guidance.

| Case | Recommended use | Historical observation, not an acceptance threshold |
| --- | --- | --- |
| `case_25` (F2FS) | Fast registration, cold-start seed, directed-corpus, and Stage-1 checks | A prior two-VM run deployed Stage 1 after about 15.5 minutes. |
| `case_21` (cfg80211) | Online training, deployment, and rfilter regression | A prior four-VM run deployed Stage 1 after about 101 minutes. |
| `case_36` (RDS) | AEC workflow and deeper target-PC reach checks | A prior guided run reached the final PC after about 31 minutes; it did not reproduce the exact crash. |

These observations came from local development runs and do not establish a
statistical speedup. A target-PC hit and an exact target crash are separate
outcomes, and neither is guaranteed by a short smoke test.

## Recommended workflow

Use [`artifact/README.md`](../artifact/README.md) for the complete, timed AEC
workflow. It uses `case_36`, the unified `qgrain/syzpilot:ndss27-ae` image, one
2-vCPU/4-GiB guest, and `procs=8`. The image supplies the SyzEncoder, matching
Syzlang metadata, build toolchains, and a disposable guest template. The
selected benchmark kernel is intentionally built locally.

To prepare another mini-benchmark case, set `CASE` to `21`, `25`, or `36` and
run the following inside a unified-image container. The Linux checkout must
contain the commit named by the selected CSV.

```bash
CASE=25
cd /root/SyzPilot

test ! -e "/root/kernels/case_${CASE}"
python experiments/compile_kernel.py \
  --workdir /root/kernels \
  --linux-git-master /root/kernels/linux-git-master \
  --config "mini-benchmark/compile_case_${CASE}.csv" -j 8

mkdir -p "/root/syzpilot-runs/case_${CASE}"
python analyzer/waypoints_extractor.py \
  -k "/root/kernels/case_${CASE}" \
  -t "benchmark/configs/case_${CASE}.title" \
  -r "benchmark/configs/case_${CASE}.report" \
  | tee "/root/syzpilot-runs/case_${CASE}/waypoints.txt"
```

Generate the manager configuration on the Fuzzer host. `BRAIN_HOST` and
`FUZZER_HOST` must be mutually reachable private addresses. The centralized
helper replaces all kernel-specific PCs and paths; it never reuses the
template's historical addresses.

```bash
export BRAIN_HOST=10.0.0.10
export FUZZER_HOST=10.0.0.20

python scripts/prepare_functional_config.py \
  --case "$CASE" \
  --waypoints-output "/root/syzpilot-runs/case_${CASE}/waypoints.txt" \
  --output "/root/syzpilot-runs/case_${CASE}/manager.cfg" \
  --brain-host "$BRAIN_HOST" \
  --fuzzer-host "$FUZZER_HOST"
```

The helper chooses a distinct manager port for each case. For `case_25`, it
also enables the versioned generic `NoGenerate` seed catalog and the bounded
directed runtime corpus. Those assets come from the pinned Syzkaller tree,
not from F2FS-specific code, a target PoC, or crash-specific values. Cases 21
and 36 retain the ordinary full-pipeline configuration without this optional
directed-corpus treatment.

Before launching a run, follow the root README to patch and build
`/root/fuzzers/SyzPilot-fuzzer`, start the Brain explicitly, and verify the
two role environments. Use a unique `-bench` filename, explicit CPU affinity,
and the same VM resources for every comparison arm.

## Functional observations

For a CPU-side smoke check, confirm Controller registration, a running QEMU
guest, increasing execution counts, accepted report-derived guidance, and
labeled Receiver batches. With two suitable GPUs, additionally check Stage-1
training, model hot deployment, and successful inference after enough positive
samples arrive. Retain useful bench, Receiver, target-hit, and crash evidence;
remove only containers and work directories created by the current run.
