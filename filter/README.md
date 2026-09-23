# Reachability Classifier and Attribution

The `filter` package contains SyzPilot's SyzEncoder-based reachability
classifier, online training pipeline, tokenizer utilities, and integrated-
gradient attribution. The Brain Controller normally invokes these components;
the commands below are useful for isolated development and diagnosis.

## Public Resources

- [SyzEncoder](https://huggingface.co/zzra1n/SyzEncoder) provides the released
  continued-pretrained encoder weights.
- [SyzPilot-dataset](https://huggingface.co/datasets/zzra1n/SyzPilot-dataset)
  provides the released research dataset.
- A compatible SyzTokenizer must use the same vocabulary and tokenization
  contract as the model being trained or replayed.

Model weights and datasets are intentionally not committed to this source
repository. Pass their local paths explicitly or use the environment variables
documented in the top-level README.

## Online Batch Format

The Receiver writes paired batch files into a task-specific data directory:

```text
/path/to/task-data/
├── progs_batch_1.pkl
├── labels_batch_1.pkl
├── progs_batch_2.pkl
├── labels_batch_2.pkl
└── ...
```

Each program must have one exactly-one-hot reachability label. Class zero is
`Unreachable`; the remaining classes represent the deepest reached waypoint.
Training and validation snapshots must be signature-disjoint. The target PoC
or reproducer is never a training input in the functional workflow.

## Stage 1: Binary Reachability

Stage 1 collapses all reached classes into one positive class while retaining
the common classifier output structure.

```bash
accelerate launch --num_processes 1 filter/train_v2.py \
  --total_steps 1000 \
  --test_interval 200 \
  --min_steps 200 \
  --patience 3 \
  --batch_size 64 \
  --grad_acc_steps 2 \
  --num_classes 5 \
  --train_stage 1 \
  --base_model_path /path/to/SyzEncoder \
  --tokenizer_path /path/to/SyzTokenizer \
  --data_dir /path/to/task-data \
  --data_idx 1,3,5,7,9,10 \
  --test_data_idx 2,4,6,8 \
  --freeze_layers \
  --is_first_train \
  --disable_wandb
```

## Stage 2: Deepest-Waypoint Classification

Stage 2 continues from an accepted Stage-1 checkpoint and learns reached-class
distinctions.

```bash
accelerate launch --num_processes 1 filter/train_v2.py \
  --total_steps 500 \
  --test_interval 100 \
  --min_steps 100 \
  --patience 2 \
  --batch_size 64 \
  --grad_acc_steps 2 \
  --num_classes 5 \
  --train_stage 2 \
  --base_model_path /path/to/SyzEncoder \
  --tokenizer_path /path/to/SyzTokenizer \
  --data_dir /path/to/task-data \
  --data_idx 1,3,5,7,9,10 \
  --test_data_idx 2,4,6,8 \
  --load_path /path/to/accepted-stage1.pt \
  --loaded_checkpoint_stage 1 \
  --freeze_layers \
  --disable_wandb
```

`--num_classes`, label order, tokenizer, waypoint chain, and maximum sequence
length must remain compatible across stages and historical-model replay. The
Controller records these contracts in training manifests and applies promotion
gates before serving a new model.

For the complete online workflow, use `brain/controller.py` as described in
the top-level [`README.md`](../README.md). For encoder pretraining, see
[`docs/pretrain_syzencoder.md`](../docs/pretrain_syzencoder.md).
