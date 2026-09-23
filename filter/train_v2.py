
import argparse
import hashlib
import json
import math
import os
import resource
import re
import sys
from datetime import datetime
from pathlib import Path
import pickle
from time import time

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from accelerate.utils import ProjectConfiguration
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import TOKEN, MODEL_MAX_LENGTH, get_model_max_length
from dataset_v2 import ProgramDatasetV2_2, load_canonical_records
from logger import TrainLogger
from model_v2 import (
    TraceClassifierV2,
    curriculum_labels,
    curriculum_logits,
)
from utils import create_collate_fn, rand_str
from common.curriculum import (
    CURRICULUM_SCHEMA_VERSION,
    curriculum_class,
    curriculum_output_classes,
)


def effective_eval_steps(num_records, eval_batch_size, requested_steps):
    """Evaluate each canonical validation record at most once per round."""
    if num_records <= 0 or eval_batch_size <= 0 or requested_steps <= 0:
        raise ValueError("evaluation sizes and step limit must be positive")
    return min(requested_steps, math.ceil(num_records / eval_batch_size))


def binary_serving_logits(raw_logits, serving_stage):
    """Project logits using the deployed stage's reachability decision."""
    if serving_stage == 1:
        return curriculum_logits(raw_logits, 1)
    if serving_stage == 2:
        if raw_logits.ndim != 2 or raw_logits.shape[1] < 2:
            raise ValueError(
                "raw logits must have shape [batch, at least 2 classes]"
            )
        return torch.stack((
            raw_logits[:, 0],
            torch.max(raw_logits[:, 1:], dim=1).values,
        ), dim=1)
    raise ValueError(f"invalid serving stage: {serving_stage}")


def write_training_manifest(save_dir, *, best_checkpoint, best_step,
                            best_eval_loss, final_step, config,
                            best_metrics=None, baseline_metrics=None):
    """Atomically publish the checkpoint selected by validation loss."""
    if not math.isfinite(float(best_eval_loss)):
        raise ValueError("best_eval_loss must be finite")
    checkpoint_path = Path(best_checkpoint).resolve()
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    validation_signature_count = int(config.get(
        "validation_signature_count", 0
    ))
    validation_signature_sha256 = str(config.get(
        "validation_signature_sha256", ""
    ))
    if (validation_signature_count <= 0 or
            re.fullmatch(r"[0-9a-f]{64}", validation_signature_sha256) is None):
        raise ValueError("validation signature fingerprint is invalid")

    manifest = {
        "schema_version": 1,
        "curriculum_schema": CURRICULUM_SCHEMA_VERSION,
        "best_checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest.hexdigest(),
        "best_step": int(best_step),
        "best_eval_loss": float(best_eval_loss),
        "final_step": int(final_step),
        "train_stage": int(config["train_stage"]),
        "num_classes": int(config["num_classes"]),
        "model_max_length": int(config.get("model_max_length", 1024)),
        "micro_batch_size": int(config.get("batch_size", 128)),
        "gradient_accumulation_steps": int(config.get("grad_acc_steps", 1)),
        "effective_batch_size": int(config.get("batch_size", 128)) *
        int(config.get("grad_acc_steps", 1)),
        "mixed_precision": str(config.get("mixed_precision", "no")),
        "training_profile": (
            "first" if config.get("is_first_train") else "continued"
        ),
        "configured_total_steps": int(config["total_steps"]),
        "test_interval": int(config["test_interval"]),
        "min_steps": int(config["min_steps"]),
        "patience": int(config["patience"]),
        "num_warmup_steps": int(config["num_warmup_steps"]),
        "assigned_physical_gpu": str(config.get(
            "assigned_physical_gpu", ""
        )),
        "initialization_wall_seconds": float(config.get(
            "initialization_wall_seconds", 0.0
        )),
        "training_wall_seconds": float(config.get(
            "training_wall_seconds", 0.0
        )),
        "trainer_process_wall_seconds": float(config.get(
            "trainer_process_wall_seconds", 0.0
        )),
        "cuda_peak_measurement_scope": str(config.get(
            "cuda_peak_measurement_scope", "unavailable"
        )),
        "max_rss_kib": int(config.get("max_rss_kib", 0)),
        "cuda_peak_allocated_bytes": int(config.get(
            "cuda_peak_allocated_bytes", 0
        )),
        "cuda_peak_reserved_bytes": int(config.get(
            "cuda_peak_reserved_bytes", 0
        )),
        "token_cache_entries": int(config.get("token_cache_entries", 0)),
        "train_token_cache": dict(config.get("train_token_cache", {})),
        "test_token_cache": dict(config.get("test_token_cache", {})),
        "train_batch_indices": list(config["data_idx"]),
        "test_batch_indices": list(config["test_data_idx"]),
        "canonical_batch_indices": list(config.get(
            "canonical_data_idx",
            list(config["data_idx"]) + list(config["test_data_idx"]),
        )),
        "test_exclude_batch_indices": list(config.get(
            "test_exclude_data_idx", config["data_idx"]
        )),
        "seen_train_batch_indices": list(config.get(
            "test_exclude_data_idx", config["data_idx"]
        )),
        "loaded_checkpoint": config.get("load_path") or None,
        "loaded_checkpoint_stage": int(config.get(
            "loaded_checkpoint_stage", 0
        )),
        "validation_signature_count": validation_signature_count,
        "validation_signature_sha256": validation_signature_sha256,
        "class_aware_replay": bool(config.get("class_aware_replay", False)),
        "replay_policy": config.get("replay_policy", "disabled"),
        "max_positive_replay": int(config.get("max_positive_replay", 1)),
        "replay_weights": {
            str(class_index): float(weight)
            for class_index, weight in config.get("replay_weights", {}).items()
        },
        "train_raw_class_counts": {
            str(class_index): int(count)
            for class_index, count in config.get(
                "train_raw_class_counts", {}
            ).items()
        },
        "train_effective_class_counts": {
            str(class_index): int(count)
            for class_index, count in config.get(
                "train_effective_class_counts", {}
            ).items()
        },
    }
    if best_metrics is not None:
        accuracy = float(best_metrics["accuracy"])
        weighted_f1 = float(best_metrics["f1_score"])
        macro_f1 = float(best_metrics["macro_f1"])
        if (not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0 or
                not math.isfinite(weighted_f1) or
                not 0.0 <= weighted_f1 <= 1.0 or
                not math.isfinite(macro_f1) or
                not 0.0 <= macro_f1 <= 1.0):
            raise ValueError("best checkpoint metrics must be finite probabilities")
        manifest["best_eval_accuracy"] = accuracy
        manifest["best_eval_weighted_f1"] = weighted_f1
        manifest["best_eval_macro_f1"] = macro_f1
    validation_class_counts = (
        best_metrics.get("class_counts") if best_metrics is not None else None
    )
    if validation_class_counts is not None:
        expected_classes = curriculum_output_classes(
            int(config["num_classes"]), int(config["train_stage"])
        )
        normalized_counts = {
            str(int(class_index)): int(count)
            for class_index, count in validation_class_counts.items()
        }
        if (set(normalized_counts) != {str(index) for index in range(expected_classes)} or
                any(count < 0 for count in normalized_counts.values())):
            raise ValueError("validation class counts do not match the curriculum stage")
        manifest["validation_class_counts"] = normalized_counts
        per_class_recall = {
            str(int(class_index)): float(recall)
            for class_index, recall in best_metrics["per_class_recall"].items()
        }
        if (set(per_class_recall) != set(normalized_counts) or
                any(not math.isfinite(recall) or not 0.0 <= recall <= 1.0
                    for recall in per_class_recall.values())):
            raise ValueError(
                "validation per-class recall does not match the curriculum stage"
            )
        manifest["validation_per_class_recall"] = per_class_recall
        binary_loss = float(best_metrics["binary_eval_loss"])
        binary_accuracy = float(best_metrics["binary_accuracy"])
        binary_macro_f1 = float(best_metrics["binary_macro_f1"])
        binary_counts = {
            str(int(class_index)): int(count)
            for class_index, count in best_metrics[
                "binary_class_counts"
            ].items()
        }
        binary_recalls = {
            str(int(class_index)): float(recall)
            for class_index, recall in best_metrics[
                "binary_per_class_recall"
            ].items()
        }
        if (not math.isfinite(binary_loss) or
                not math.isfinite(binary_accuracy) or
                not 0.0 <= binary_accuracy <= 1.0 or
                not math.isfinite(binary_macro_f1) or
                not 0.0 <= binary_macro_f1 <= 1.0 or
                set(binary_counts) != {"0", "1"} or
                any(count < 0 for count in binary_counts.values()) or
                set(binary_recalls) != {"0", "1"} or
                any(not math.isfinite(recall) or not 0.0 <= recall <= 1.0
                    for recall in binary_recalls.values())):
            raise ValueError("binary checkpoint metrics are invalid")
        manifest.update({
            "binary_eval_loss": binary_loss,
            "binary_eval_accuracy": binary_accuracy,
            "binary_eval_macro_f1": binary_macro_f1,
            "binary_validation_class_counts": binary_counts,
            "binary_validation_per_class_recall": binary_recalls,
        })
    loaded_checkpoint = config.get("load_path")
    if loaded_checkpoint:
        loaded_digest = hashlib.sha256()
        with Path(loaded_checkpoint).resolve().open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                loaded_digest.update(chunk)
        manifest["loaded_checkpoint_sha256"] = loaded_digest.hexdigest()
    if baseline_metrics is not None:
        baseline_accuracy = float(baseline_metrics["accuracy"])
        baseline_macro_f1 = float(baseline_metrics["macro_f1"])
        baseline_loss = float(baseline_metrics["eval_loss"])
        if (not math.isfinite(baseline_loss) or
                not math.isfinite(baseline_accuracy) or
                not 0.0 <= baseline_accuracy <= 1.0 or
                not math.isfinite(baseline_macro_f1) or
                not 0.0 <= baseline_macro_f1 <= 1.0):
            raise ValueError("baseline metrics must be finite and bounded")
        baseline_counts = {
            str(int(class_index)): int(count)
            for class_index, count in baseline_metrics["class_counts"].items()
        }
        baseline_recalls = {
            str(int(class_index)): float(recall)
            for class_index, recall in baseline_metrics[
                "per_class_recall"
            ].items()
        }
        if (validation_class_counts is None or
                baseline_counts != manifest["validation_class_counts"] or
                set(baseline_recalls) != set(baseline_counts) or
                any(not math.isfinite(recall) or not 0.0 <= recall <= 1.0
                    for recall in baseline_recalls.values())):
            raise ValueError(
                "baseline metrics do not match the validation class schema"
            )
        manifest.update({
            "baseline_eval_loss": baseline_loss,
            "baseline_eval_accuracy": baseline_accuracy,
            "baseline_eval_macro_f1": baseline_macro_f1,
            "baseline_validation_class_counts": baseline_counts,
            "baseline_validation_per_class_recall": baseline_recalls,
        })
        baseline_binary_loss = float(
            baseline_metrics["binary_eval_loss"]
        )
        baseline_binary_accuracy = float(
            baseline_metrics["binary_accuracy"]
        )
        baseline_binary_macro_f1 = float(
            baseline_metrics["binary_macro_f1"]
        )
        baseline_binary_counts = {
            str(int(class_index)): int(count)
            for class_index, count in baseline_metrics[
                "binary_class_counts"
            ].items()
        }
        baseline_binary_recalls = {
            str(int(class_index)): float(recall)
            for class_index, recall in baseline_metrics[
                "binary_per_class_recall"
            ].items()
        }
        if (not math.isfinite(baseline_binary_loss) or
                not math.isfinite(baseline_binary_accuracy) or
                not 0.0 <= baseline_binary_accuracy <= 1.0 or
                not math.isfinite(baseline_binary_macro_f1) or
                not 0.0 <= baseline_binary_macro_f1 <= 1.0 or
                baseline_binary_counts != binary_counts or
                set(baseline_binary_recalls) != {"0", "1"} or
                any(not math.isfinite(recall) or not 0.0 <= recall <= 1.0
                    for recall in baseline_binary_recalls.values())):
            raise ValueError("baseline binary metrics are invalid")
        manifest.update({
            "baseline_binary_eval_loss": baseline_binary_loss,
            "baseline_binary_eval_accuracy": baseline_binary_accuracy,
            "baseline_binary_eval_macro_f1": baseline_binary_macro_f1,
            "baseline_binary_validation_class_counts": (
                baseline_binary_counts
            ),
            "baseline_binary_validation_per_class_recall": (
                baseline_binary_recalls
            ),
        })
    manifest_path = Path(save_dir) / "training_manifest.json"
    temp_path = manifest_path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")
        file_handle.flush()
        os.fsync(file_handle.fileno())
    os.replace(temp_path, manifest_path)
    return manifest_path


class TrainerV2:
    def __init__(self, config):
        # Initialize dataset and get label dimension
        self.t0 = time()
        if not config.get("session_name"):
            config["session_name"] = (
                f"TraceClassifier-v2.0-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        config["save_dir"] = Path(config["log_dir"]) / config["session_name"]
        config["save_dir"].mkdir(parents=True, exist_ok=True)
        config["data_idx"] = list(map(int, config["data_idx"].split(","))) if config["data_idx"] else []
        config["test_data_idx"] = list(map(int, config["test_data_idx"].split(","))) if config["test_data_idx"] else []
        config["canonical_data_idx"] = (
            list(map(int, config["canonical_data_idx"].split(",")))
            if config["canonical_data_idx"] else
            config["data_idx"] + config["test_data_idx"]
        )
        config["test_exclude_data_idx"] = (
            list(map(int, config["test_exclude_data_idx"].split(",")))
            if config["test_exclude_data_idx"] else list(config["data_idx"])
        )
        self.ckpt_list = {}
        self.best_step = 0

        self.config = config
        self.start_time = time()
        if self.config["grad_acc_steps"] <= 0:
            raise ValueError("grad_acc_steps must be positive")

        kwargs = DDPK(find_unused_parameters=True)
        if config["disable_wandb"]:
            self.accelerator = Accelerator(
                log_with="tensorboard",
                gradient_accumulation_steps=config["grad_acc_steps"],
                kwargs_handlers=[kwargs],
                project_config=ProjectConfiguration(
                    project_dir=config["log_dir"],
                    # logging_dir=config["save_dir"],
                ),
            )
            self.accelerator.init_trackers(project_name=config["session_name"])
            # if self.accelerator.is_main_process:
            #     self.tb_writer = SummaryWriter(log_dir=config["save_dir"])
        else:
            self.accelerator = Accelerator(
                log_with="wandb",
                gradient_accumulation_steps=config["grad_acc_steps"],
                kwargs_handlers=[kwargs],
            )
            self.accelerator.init_trackers(
                project_name=config["session_name"],
                config=config,
                init_kwargs={
                    "wandb": {
                        "name": config["run_name"],
                        "mode": "online",
                        "entity": config["wandb_entity"],
                    }
                },
            )
        self.config["mixed_precision"] = self.accelerator.mixed_precision
        if self.accelerator.device.type == "cuda":
            # Include model construction, checkpoint restore, dataloaders, and
            # the training loop in the allocator peak recorded in the manifest.
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)
            self.config["cuda_peak_measurement_scope"] = (
                "accelerator_initialized_through_training_completion"
            )

        # Initialize model
        self.model = TraceClassifierV2(
            self.config["base_model_path"],
            self.config["num_classes"],
            stage=self.config["train_stage"],
        )
        if config["load_path"]:
            loaded_weights = torch.load(config["load_path"], weights_only=True)
            self.model.load_state_dict(loaded_weights)
        if config["freeze_layers"]:
            import re
            encoder_layer_id_pattern = re.compile("encoder.layer.(\d+)")
            for name, param in self.model.named_parameters():
                if name.startswith("base_model.embeddings"):
                    param.requires_grad_(False)
                elif name.startswith("base_model.encoder.layer"):
                    layer_id = eval(re.findall(encoder_layer_id_pattern, name)[0])
                    if layer_id < 10:
                        param.requires_grad_(False)
        # Enable gradient checkpointing to reduce activation memory (~3-4GB savings)
        if hasattr(self.model.base_model, "gradient_checkpointing_enable"):
            self.model.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["tokenizer_path"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model_max_length = get_model_max_length(self.config["base_model_path"])
        self.config["model_max_length"] = model_max_length

        # Load dataset
        canonical_indices = config["canonical_data_idx"]
        train_ds = ProgramDatasetV2_2(
            config["data_dir"],
            config["num_classes"],
            config["data_idx"],
            canonical_indices=canonical_indices,
            class_aware_replay=config["train_stage"] == 2,
        )
        self.config["class_aware_replay"] = train_ds.class_aware_replay
        self.config["replay_policy"] = (
            "sqrt_inverse_frequency_capped" if train_ds.class_aware_replay
            else "disabled"
        )
        self.config["max_positive_replay"] = train_ds.max_positive_replay
        self.config["replay_weights"] = dict(train_ds.replay_weights)
        self.config["train_raw_class_counts"] = dict(train_ds.raw_class_counts)
        self.config["train_effective_class_counts"] = dict(
            train_ds.effective_class_counts
        )
        self.train_collator = create_collate_fn(
            self.tokenizer,
            max_length=model_max_length,
            cache_entries=self.config["token_cache_entries"],
        )
        self.train_dl = DataLoader(
            train_ds,
            batch_size=self.config["batch_size"],
            collate_fn=self.train_collator,
        )
        test_exclude_records = load_canonical_records(
            config["data_dir"],
            config["num_classes"],
            config["test_exclude_data_idx"],
        )
        test_ds = ProgramDatasetV2_2(
            config["data_dir"],
            config["num_classes"],
            config["test_data_idx"],
            exclude_signatures=test_exclude_records,
            canonical_indices=canonical_indices,
            repeat=False,
        )
        validation_class_counts = {
            class_index: 0
            for class_index in range(curriculum_output_classes(
                config["num_classes"], config["train_stage"]
            ))
        }
        for _, _, exact_class in test_ds.records.values():
            grouped_class = curriculum_class(
                exact_class, config["num_classes"], config["train_stage"]
            )
            validation_class_counts[grouped_class] += 1
        self.config["validation_dataset_class_counts"] = validation_class_counts
        validation_signatures = sorted(test_ds.signatures)
        self.config["validation_signature_count"] = len(validation_signatures)
        self.config["validation_signature_sha256"] = hashlib.sha256(
            "\n".join(validation_signatures).encode("utf-8")
        ).hexdigest()
        full_eval_steps = math.ceil(
            len(test_ds.records) / (4 * self.config["batch_size"])
        )
        # Promotion evidence must cover the complete signature-disjoint
        # holdout. Test-only diagnostics may retain an explicit step cap.
        requested_eval_steps = (
            self.config["test_steps"]
            if self.config["test_only"] else full_eval_steps
        )
        self.eval_steps = effective_eval_steps(
            len(test_ds.records),
            4 * self.config["batch_size"],
            requested_eval_steps,
        )
        self.config["effective_test_steps"] = self.eval_steps
        self.test_collator = create_collate_fn(
            self.tokenizer,
            max_length=model_max_length,
            cache_entries=self.config["token_cache_entries"],
        )
        self.test_dl = DataLoader(
            test_ds,
            batch_size=4 * self.config["batch_size"],
            collate_fn=self.test_collator,
        )
        self.test_it = iter(self.test_dl)

        # Optimizer — only optimize trainable parameters (saves ~1.9GB on frozen layers)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        # Learning rate scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            self.config["num_warmup_steps"] * self.accelerator.num_processes,
            self.config["total_steps"] * self.accelerator.num_processes,
        )

        # Prepare with accelerate
        self.model, self.optimizer, self.train_dl, self.scheduler, self.test_dl = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dl, self.scheduler, self.test_dl
        )
        self.config["initialization_wall_seconds"] = time() - self.t0

        if self.accelerator.is_main_process:
            self.logger = TrainLogger(config["save_dir"], config["session_name"])
            self.logger.info("Config:")
            for _k, _v in config.items():
                self.logger.info(f'''  {_k} -> {_v}''')

    @torch.no_grad()
    def test(self, binary_serving_stage=None):
        self.model.eval()
        if binary_serving_stage is None:
            binary_serving_stage = self.config["train_stage"]
        total_loss = 0
        total_binary_loss = 0
        num_batches = 0
        num_examples = 0

        prog_bar = tqdm(
            range(self.eval_steps),
            desc="Testing",
            disable=not self.accelerator.is_main_process,
            ncols=100,
        )
        y_test = []
        y_pred = []
        binary_y_test = []
        binary_y_pred = []
        self.test_it = iter(self.test_dl)  # Reset iterator for reproducibility
        while True:
            input_ids, attention_mask, labels = next(self.test_it)

            raw_logits = self.model(input_ids, attention_mask)
            exact_labels = torch.argmax(labels, dim=1)
            objective_logits = curriculum_logits(
                raw_logits, self.config["train_stage"]
            )
            objective_labels = curriculum_labels(
                exact_labels,
                self.config["num_classes"],
                self.config["train_stage"],
            )
            loss = torch.nn.functional.cross_entropy(
                objective_logits, objective_labels
            )
            binary_logits = binary_serving_logits(
                raw_logits, binary_serving_stage
            )
            binary_labels = curriculum_labels(
                exact_labels, self.config["num_classes"], 1
            )
            binary_loss = torch.nn.functional.cross_entropy(
                binary_logits, binary_labels
            )
            if (not torch.isfinite(loss).all() or
                    not torch.isfinite(binary_loss).all()):
                raise FloatingPointError("non-finite evaluation loss")
            pred_labels = torch.argmax(objective_logits, dim=1)
            binary_pred_labels = torch.argmax(binary_logits, dim=1)
            batch_examples = int(objective_labels.shape[0])
            total_loss += loss.item() * batch_examples
            total_binary_loss += binary_loss.item() * batch_examples
            num_examples += batch_examples
            num_batches += 1
            y_test.append(objective_labels.cpu().numpy())
            y_pred.append(pred_labels.cpu().numpy())
            binary_y_test.append(binary_labels.cpu().numpy())
            binary_y_pred.append(binary_pred_labels.cpu().numpy())

            prog_bar.update(1)
            if num_batches >= self.eval_steps:
                prog_bar.close()
                break

        y_test = np.concatenate(y_test)
        y_pred = np.concatenate(y_pred)
        binary_y_test = np.concatenate(binary_y_test)
        binary_y_pred = np.concatenate(binary_y_pred)
        if self.config["test_only"] and self.accelerator.is_main_process:
            with open(self.config["save_dir"] / "test_labels.pkl", "wb") as fp:
                pickle.dump((y_test, y_pred), fp)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0,
        )
        expected_classes = curriculum_output_classes(
            self.config["num_classes"], self.config["train_stage"]
        )
        _, per_class_recall, _, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=list(range(expected_classes)),
            average=None,
            zero_division=0,
        )
        _, _, macro_f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=list(range(expected_classes)),
            average="macro",
            zero_division=0,
        )
        avg_loss = total_loss / max(num_examples, 1)
        binary_accuracy = accuracy_score(binary_y_test, binary_y_pred)
        _, binary_per_class_recall, _, _ = precision_recall_fscore_support(
            binary_y_test,
            binary_y_pred,
            labels=[0, 1],
            average=None,
            zero_division=0,
        )
        _, _, binary_macro_f1, _ = precision_recall_fscore_support(
            binary_y_test,
            binary_y_pred,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
        log_info = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "macro_f1": macro_f1,
            "eval_loss": avg_loss,
            "class_counts": {
                class_index: int(np.sum(y_test == class_index))
                for class_index in range(expected_classes)
            },
            "per_class_recall": {
                class_index: float(per_class_recall[class_index])
                for class_index in range(expected_classes)
            },
            "binary_eval_loss": (
                total_binary_loss / max(num_examples, 1)
            ),
            "binary_accuracy": binary_accuracy,
            "binary_macro_f1": binary_macro_f1,
            "binary_class_counts": {
                class_index: int(np.sum(binary_y_test == class_index))
                for class_index in range(2)
            },
            "binary_per_class_recall": {
                class_index: float(binary_per_class_recall[class_index])
                for class_index in range(2)
            },
        }

        return accuracy, time(), log_info

    def save(self, step: int):
        if not self.accelerator.is_main_process:
            return
        states = self.accelerator.unwrap_model(self.model).state_dict()
        save_path = self.config["save_dir"] / f"step-{step}.pt"
        torch.save(states, save_path)
        self.ckpt_list[step] = save_path
        while len(self.ckpt_list) > 3:
            removable = [saved_step for saved_step in self.ckpt_list
                         if saved_step != self.best_step]
            if not removable:
                break
            oldest_step = min(removable)
            self.ckpt_list[oldest_step].unlink()
            del self.ckpt_list[oldest_step]

    def run(self):
        step = 0
        runtime_log = {"configs": self.config}
        start_time = time()
        total_step = self.config["total_steps"]
        test_interval = self.config["test_interval"]
        save_interval = self.config["save_interval"]
        min_steps = self.config.get("min_steps", 200)
        patience = self.config.get("patience", 3)

        baseline_serving_stage = (
            int(self.config.get("loaded_checkpoint_stage", 0)) or
            self.config["train_stage"]
        )
        acc, current_time, log_info = self.test(
            binary_serving_stage=baseline_serving_stage
        )
        baseline_eval_metrics = (
            dict(log_info) if self.config.get("load_path") else None
        )
        if self.accelerator.is_main_process:
            print('Calculating initial stats')
            self.accelerator.print(f"Initial accuracy: {acc:0.6f}")
            spent_time = current_time - start_time
            to_log = {
                "test/accuracy": acc,
                "test/precision": log_info["precision"],
                "test/recall": log_info["recall"],
                "test/f1_score": log_info["f1_score"],
                "test/time": "%.2f"%spent_time
            }
            runtime_log[step] = to_log
            self.accelerator.log(to_log, 0)
        if self.config["test_only"]:
            return

        # Early stopping state
        best_eval_loss = float('inf')
        best_eval_metrics = None
        patience_counter = 0
        best_step = 0

        train_iterator = iter(self.train_dl)
        prog_bar = tqdm(
            total=total_step,
            ncols=100,
            disable=not self.accelerator.is_main_process,
        )
        while step < total_step:
            try:
                input_ids, attention_mask, labels = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_dl)
                input_ids, attention_mask, labels = next(train_iterator)
            self.model.train()

            with self.accelerator.accumulate(self.model):
                loss, _, _, _ = self.model(
                    input_ids, attention_mask, labels, output="train"
                )
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(
                        f"non-finite training loss before step {step + 1}"
                    )
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), 1.0
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if not self.accelerator.sync_gradients:
                continue
            step += 1

            prog_bar.set_description(f"Step: {step}, Loss: {loss:0.3f}, LR: {self.optimizer.param_groups[0]['lr']:0.3e}")
            self.accelerator.log(
                {
                    "train/loss": float(loss),
                    "train/LR": float(self.scheduler.get_last_lr()[0]),
                    "train/time_minutes": (time() - start_time) // 60,
                },
                step=step,
            )
            prog_bar.update(1)

            if step % test_interval == 0:
                eval_acc, current_time, eval_info = self.test()
                eval_loss = eval_info["eval_loss"]
                if not math.isfinite(eval_loss):
                    raise FloatingPointError(
                        f"non-finite evaluation loss at step {step}"
                    )
                if self.accelerator.is_main_process:
                    self.accelerator.print(f"Step: {step}, Test accuracy: {eval_acc:0.3f}, eval_loss: {eval_loss:0.4f}")
                    spent_time = current_time - start_time
                    to_log = {
                        "test/loss": eval_loss,
                        "test/LR": float(self.scheduler.get_last_lr()[0]),
                        "test/accuracy": eval_acc,
                        "test/precision": eval_info["precision"],
                        "test/recall": eval_info["recall"],
                        "test/f1_score": eval_info["f1_score"],
                        "test/macro_f1": eval_info["macro_f1"],
                        "test/time": "%.2f"%spent_time
                    }
                    runtime_log[step] = to_log
                    self.accelerator.log(to_log, step)

                    # Early stopping: eval_loss lower is better
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        patience_counter = 0
                        best_step = step
                        self.best_step = step
                        best_eval_metrics = dict(eval_info)
                        self.save(step)  # Save best model
                    elif step >= min_steps:
                        patience_counter += 1
                        if patience_counter >= patience:
                            self.accelerator.print(
                                f"Early stopping at step {step} (best at step {best_step}, "
                                f"eval_loss={best_eval_loss:0.4f}, patience={patience})"
                            )
                            break

            if step % save_interval == 0:
                self.save(step)

        # Keep the final checkpoint for diagnostics, but deploy the validation
        # winner recorded in the manifest below.
        if best_step == 0:
            raise RuntimeError("training finished without a validation checkpoint")
        elif step != best_step:
            self.save(step)

        if self.accelerator.is_main_process:
            if self.accelerator.device.type == "cuda":
                torch.cuda.synchronize(self.accelerator.device)
                self.config["cuda_peak_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated(self.accelerator.device)
                )
                self.config["cuda_peak_reserved_bytes"] = (
                    torch.cuda.max_memory_reserved(self.accelerator.device)
                )
            self.config["training_wall_seconds"] = time() - start_time
            self.config["trainer_process_wall_seconds"] = time() - self.t0
            self.config["max_rss_kib"] = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            self.config["train_token_cache"] = self.train_collator.cache_info()
            self.config["test_token_cache"] = self.test_collator.cache_info()
            best_checkpoint = self.ckpt_list.get(best_step)
            if best_checkpoint is None or not best_checkpoint.is_file():
                raise RuntimeError(f"best checkpoint for step {best_step} is missing")
            manifest_path = write_training_manifest(
                self.config["save_dir"],
                best_checkpoint=best_checkpoint,
                best_step=best_step,
                best_eval_loss=best_eval_loss,
                final_step=step,
                config=self.config,
                best_metrics=best_eval_metrics,
                baseline_metrics=baseline_eval_metrics,
            )
            self.accelerator.print(f"Training manifest saved to {manifest_path}")

        prog_bar.close()
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train the reach filter')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--total_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--train_stage', type=int, choices=(1, 2), default=1)

    parser.add_argument('--base_model_path', type=str, default='/opt/syzpilot/models/customized_starencoder_5w/final_syzencoder/')
    parser.add_argument('--tokenizer_path', type=str, default='/opt/syzpilot/models/customized_tokenizer_224w/')
    parser.add_argument('--freeze_layers', action='store_true')

    parser.add_argument('--T', type=float, default=0.07)
    parser.add_argument('--learning_rate', type=float, default=1.0e-5)
    parser.add_argument('--weight_decay', type=float, default=1.0e-4)
    parser.add_argument('--num_warmup_steps', type=int, default=50)
    parser.add_argument('--grad_acc_steps', type=int, default=1)
    parser.add_argument(
        '--assigned_physical_gpu', type=str, default='',
        help='Controller-selected physical GPU index for manifest auditing',
    )
    parser.add_argument(
        '--token_cache_entries', type=int, default=50000,
        help='Maximum exact process-local token encodings to retain; 0 disables',
    )

    parser.add_argument('--data_dir', type=str, default='/artifact/datasets/test_only/test_num_class_5/')
    parser.add_argument('--data_idx', type=str, default='')
    parser.add_argument('--test_data_idx', type=str, default='')
    parser.add_argument('--canonical_data_idx', type=str, default='')
    parser.add_argument('--test_exclude_data_idx', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--session_name', type=str, default='')

    parser.add_argument('--save_interval', type=float, default=500)
    parser.add_argument('--test_interval', type=float, default=500)
    parser.add_argument('--test_steps', type=int, default=25)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument(
        '--loaded_checkpoint_stage', type=int, choices=(0, 1, 2), default=0
    )
    # Adaptive early stopping
    parser.add_argument('--min_steps', type=int, default=200, help='Minimum training steps before early stopping')
    parser.add_argument('--patience', type=int, default=3, help='Stop after N consecutive evals without improvement')
    parser.add_argument('--is_first_train', action='store_true', help='First training (from scratch) vs continued training')

    # parser.add_argument('--trainset_rate', type=float, default=0.9)
    # parser.add_argument('--pos_weight', type=float, nargs='+', default=None)

    parser.add_argument('--disable_wandb', action='store_true', default=True)
    # parser.add_argument('--proj_name', type=str, default='default_wandb_project')
    # parser.add_argument('--run_name', type=str, default='default_wandb_run')
    # parser.add_argument('--wandb_entity', type=str, default='KernelAI')

    args = parser.parse_args()
    config = vars(args)

    trainer = TrainerV2(config)
    trainer.run()
