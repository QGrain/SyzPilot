# =============================================
# file: server/controller_config.py
# A single place to tune the controller behavior.
# =============================================
import os
import math
from dataclasses import dataclass, field
from pathlib import Path

# Get the brain directory (where this config file is located)
BRAIN_DIR = Path(__file__).parent.absolute()

TOKEN = ''


def _path_roots_from_env(name: str, defaults: tuple) -> tuple:
    """Read a platform-separated list of trusted local directory roots."""
    value = os.getenv(name)
    if value is None:
        return defaults
    return tuple(part.strip() for part in value.split(os.pathsep) if part.strip())


def get_model_max_length(base_model):
    if 'codebert-base' in base_model:
        return 512
    elif 'starencoder' in base_model:
        return 1024
    else:
        return 1024


@dataclass
class ControllerConfig:
    # Dashboard configuration
    dashboard_token: str = os.getenv(
        "DASHBOARD_TOKEN", "syzpilot-local-dev-token"
    )
    dashboard_enabled: bool = True
    direct_only: bool = field(default_factory=lambda: os.getenv(
        "SYZPILOT_DIRECT_ONLY", "false"
    ).strip().lower() in ("1", "true", "yes"))

    # Logging
    log_dir: str = str(BRAIN_DIR / "logs")

    # ---- Data & training ----
    data_root: str = str(BRAIN_DIR / "receiver_data")   # where shards and merged PKLs live (absolute path)
    min_samples_to_train: int = 1000                      # trigger a train run after this many new samples (matches receiver stage1_threshold)
    training_warmup_seconds: int = int(os.getenv(
        "SYZPILOT_TRAINING_WARMUP_SECONDS", "1800"
    ))
    base_model: str = field(default_factory=lambda: os.getenv(
        "SYZPILOT_BASE_MODEL_PATH",
        "/opt/syzpilot/models/SyzEncoder_224w_full/best_model/",
    ))
    tokenizer: str = field(default_factory=lambda: os.getenv(
        "SYZPILOT_TOKENIZER_PATH",
        "/artifact/assets/models/SyzTokenizer_224w/",
    ))
    epochs: float = 0.2                                    # short online run
    # A 64-sample micro-batch with two-step accumulation preserves the
    # effective batch of 128 while substantially reducing peak VRAM.
    batch_size: int = int(os.getenv("SYZPILOT_TRAINING_BATCH_SIZE", "64"))
    grad_acc_steps: int = int(os.getenv(
        "SYZPILOT_TRAINING_GRAD_ACC_STEPS", "2"
    ))
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    num_warmup_steps: int = int(os.getenv(
        "SYZPILOT_TRAINING_WARMUP_STEPS", "100"
    ))
    first_train_total_steps: int = int(os.getenv(
        "SYZPILOT_FIRST_TRAIN_TOTAL_STEPS", "1000"
    ))
    first_train_test_interval: int = int(os.getenv(
        "SYZPILOT_FIRST_TRAIN_TEST_INTERVAL", "200"
    ))
    first_train_min_steps: int = int(os.getenv(
        "SYZPILOT_FIRST_TRAIN_MIN_STEPS", "200"
    ))
    first_train_patience: int = int(os.getenv(
        "SYZPILOT_FIRST_TRAIN_PATIENCE", "3"
    ))
    continued_train_total_steps: int = int(os.getenv(
        "SYZPILOT_CONTINUED_TRAIN_TOTAL_STEPS", "500"
    ))
    continued_train_test_interval: int = int(os.getenv(
        "SYZPILOT_CONTINUED_TRAIN_TEST_INTERVAL", "100"
    ))
    continued_train_min_steps: int = int(os.getenv(
        "SYZPILOT_CONTINUED_TRAIN_MIN_STEPS", "100"
    ))
    continued_train_patience: int = int(os.getenv(
        "SYZPILOT_CONTINUED_TRAIN_PATIENCE", "2"
    ))
    trainset_rate: float = 0.9
    disable_wandb: bool = True
    attribution_min_accuracy: float = 0.85
    promotion_majority_margin: float = float(os.getenv(
        "SYZPILOT_PROMOTION_MAJORITY_MARGIN", "0.05"
    ))
    promotion_stage1_min_support: int = int(os.getenv(
        "SYZPILOT_PROMOTION_STAGE1_MIN_SUPPORT", "32"
    ))
    promotion_stage1_min_macro_f1: float = float(os.getenv(
        "SYZPILOT_PROMOTION_STAGE1_MIN_MACRO_F1", "0.75"
    ))
    promotion_stage1_min_recall: float = float(os.getenv(
        "SYZPILOT_PROMOTION_STAGE1_MIN_RECALL", "0.75"
    ))
    promotion_stage2_min_support: int = int(os.getenv(
        "SYZPILOT_PROMOTION_STAGE2_MIN_SUPPORT", "16"
    ))
    promotion_stage2_min_macro_f1: float = float(os.getenv(
        "SYZPILOT_PROMOTION_STAGE2_MIN_MACRO_F1", "0.60"
    ))
    promotion_stage2_min_unreachable_recall: float = float(os.getenv(
        "SYZPILOT_PROMOTION_STAGE2_MIN_UNREACHABLE_RECALL", "0.75"
    ))
    promotion_stage2_min_reached_recall: float = float(os.getenv(
        "SYZPILOT_PROMOTION_STAGE2_MIN_REACHED_RECALL", "0.40"
    ))
    promotion_stage2_min_final_recall: float = float(os.getenv(
        "SYZPILOT_PROMOTION_STAGE2_MIN_FINAL_RECALL", "0.50"
    ))
    clf: str = "fc"                                        # or "lstm"
    # Online trainers are single-process and use DataLoader(num_workers=0), so
    # bounded Rayon parallelism speeds up exact tokenization without forked
    # tokenizer pools.  Keep one training process per physical GPU to avoid
    # throughput collapse from colocated 1024-token batches.
    training_gpu_ids: tuple = tuple(
        gpu_id.strip()
        for gpu_id in os.getenv("SYZPILOT_TRAINING_GPU_IDS", "0").split(",")
        if gpu_id.strip()
    )
    # Fallback devices are considered only after every eligible primary
    # device. Keep this empty in production; functional runs may explicitly
    # share the attribution device through the controller's physical lease.
    training_fallback_gpu_ids: tuple = tuple(
        gpu_id.strip()
        for gpu_id in os.getenv(
            "SYZPILOT_TRAINING_FALLBACK_GPU_IDS", ""
        ).split(",")
        if gpu_id.strip()
    )
    attribution_gpu_id: str = os.getenv("SYZPILOT_ATTRIBUTION_GPU_ID", "1")
    inference_gpu_id: str = os.getenv("SYZPILOT_INFERENCE_GPU_ID", "1")
    attribution_internal_batch_size: int = int(os.getenv(
        "SYZPILOT_ATTRIBUTION_INTERNAL_BATCH_SIZE", "5"
    ))
    attribution_slots: int = int(os.getenv(
        "SYZPILOT_ATTRIBUTION_SLOTS", "1"
    ))
    training_slots_per_gpu: int = int(os.getenv(
        "SYZPILOT_TRAINING_SLOTS_PER_GPU", "1"
    ))
    training_min_free_mib: int = int(os.getenv(
        "SYZPILOT_TRAINING_MIN_FREE_MIB", "24000"
    ))
    training_max_gpu_utilization: int = int(os.getenv(
        "SYZPILOT_TRAINING_MAX_GPU_UTILIZATION", "20"
    ))
    training_gpu_probe_samples: int = int(os.getenv(
        "SYZPILOT_TRAINING_GPU_PROBE_SAMPLES", "3"
    ))
    training_gpu_probe_interval_seconds: float = float(os.getenv(
        "SYZPILOT_TRAINING_GPU_PROBE_INTERVAL_SECONDS", "1.0"
    ))
    # Receiver retries durable requests every five seconds. A longer lease
    # preserves FIFO position across normal HTTP/probe jitter while allowing a
    # dead Receiver or terminal preflight failure to stop blocking the queue.
    training_waiter_lease_seconds: float = float(os.getenv(
        "SYZPILOT_TRAINING_WAITER_LEASE_SECONDS", "30.0"
    ))
    tokenizer_rayon_threads: int = int(os.getenv(
        "SYZPILOT_TOKENIZER_THREADS", "8"
    ))
    token_cache_entries: int = int(os.getenv(
        "SYZPILOT_TOKEN_CACHE_ENTRIES", "50000"
    ))

    # ---- Static analysis ----
    report_paths: dict = None  # task_name → crash report path for PathBasedAnalyzer
    target_funcs: dict = None  # task_name → KallGraph target function
    kallgraph_dirs: dict = None  # task_name → KallGraph output directory
    syzkaller_syslinux: str = field(default_factory=lambda: os.getenv(
        "SYZPILOT_SYZKALLER_SYSLINUX",
        "/artifact/assets/syzlang/sys/linux/",
    ))
    guidance_report_roots: tuple = field(default_factory=lambda:
        _path_roots_from_env(
            "SYZPILOT_GUIDANCE_REPORT_ROOTS",
            ("/artifact/assets",),
        ))
    guidance_kallgraph_roots: tuple = field(default_factory=lambda:
        _path_roots_from_env(
            "SYZPILOT_GUIDANCE_KALLGRAPH_ROOTS",
            ("/artifact/assets/kallgraph",),
        ))
    guidance_max_report_bytes: int = 8 * 1024 * 1024
    guidance_max_callgraph_bytes: int = 512 * 1024 * 1024
    syzlang_manifest_path: str = os.getenv(
        "SYZPILOT_SYZLANG_MANIFEST",
        "/artifact/assets/syzlang/linux-amd64.json",
    )
    syzlang_manifest_max_bytes: int = int(os.getenv(
        "SYZPILOT_SYZLANG_MANIFEST_MAX_BYTES", str(64 * 1024 * 1024)
    ))

    def __post_init__(self):
        if self.syzlang_manifest_max_bytes <= 0:
            raise ValueError("syzlang_manifest_max_bytes must be positive")
        if not self.training_gpu_ids:
            raise ValueError("at least one training GPU must be configured")
        canonical_gpu_ids = all(
            gpu_id.isascii() and gpu_id.isdecimal() and
            gpu_id == str(int(gpu_id))
            for gpu_id in self.training_gpu_ids
        )
        if (not canonical_gpu_ids or
                len(set(self.training_gpu_ids)) != len(self.training_gpu_ids)):
            raise ValueError(
                "training_gpu_ids must contain unique numeric GPU indices"
            )
        canonical_fallback_gpu_ids = all(
            gpu_id.isascii() and gpu_id.isdecimal() and
            gpu_id == str(int(gpu_id))
            for gpu_id in self.training_fallback_gpu_ids
        )
        if (not canonical_fallback_gpu_ids or
                len(set(self.training_fallback_gpu_ids)) !=
                len(self.training_fallback_gpu_ids)):
            raise ValueError(
                "training_fallback_gpu_ids must contain unique numeric GPU "
                "indices"
            )
        if set(self.training_gpu_ids).intersection(
                self.training_fallback_gpu_ids):
            raise ValueError(
                "primary and fallback training GPU indices must not overlap"
            )
        canonical_attribution_gpu = (
            self.attribution_gpu_id.isascii() and
            self.attribution_gpu_id.isdecimal() and
            self.attribution_gpu_id == str(int(self.attribution_gpu_id))
        )
        if (not canonical_attribution_gpu or
                self.attribution_gpu_id in self.training_gpu_ids):
            raise ValueError(
                "attribution_gpu_id must be numeric and absent from primary "
                "training GPUs"
            )
        canonical_inference_gpu = (
            self.inference_gpu_id.isascii() and
            self.inference_gpu_id.isdecimal() and
            self.inference_gpu_id == str(int(self.inference_gpu_id))
        )
        if (not canonical_inference_gpu or
                self.inference_gpu_id in self.training_gpu_ids):
            raise ValueError(
                "inference_gpu_id must be numeric and absent from primary "
                "training GPUs"
            )
        if self.inference_gpu_id != self.attribution_gpu_id:
            raise ValueError(
                "inference_gpu_id and attribution_gpu_id must match so "
                "SyzPilot uses only one non-training GPU"
            )
        if self.attribution_internal_batch_size <= 0:
            raise ValueError(
                "attribution_internal_batch_size must be positive"
            )
        if self.attribution_slots != 1:
            raise ValueError(
                "attribution_slots must be 1 while attribution mutates "
                "process-global module state"
            )
        if self.training_slots_per_gpu != 1:
            raise ValueError(
                "training_slots_per_gpu must be 1 while training/export and "
                "attribution share physical GPU leases"
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.grad_acc_steps <= 0:
            raise ValueError("grad_acc_steps must be positive")
        from model_promotion import PromotionThresholds
        PromotionThresholds(
            majority_margin=self.promotion_majority_margin,
            stage1_min_support=self.promotion_stage1_min_support,
            stage1_min_macro_f1=self.promotion_stage1_min_macro_f1,
            stage1_min_recall=self.promotion_stage1_min_recall,
            stage2_min_support=self.promotion_stage2_min_support,
            stage2_min_macro_f1=self.promotion_stage2_min_macro_f1,
            stage2_min_unreachable_recall=(
                self.promotion_stage2_min_unreachable_recall
            ),
            stage2_min_reached_recall=(
                self.promotion_stage2_min_reached_recall
            ),
            stage2_min_final_recall=self.promotion_stage2_min_final_recall,
        ).validate()
        training_profiles = {
            "first": (
                self.first_train_total_steps,
                self.first_train_test_interval,
                self.first_train_min_steps,
                self.first_train_patience,
            ),
            "continued": (
                self.continued_train_total_steps,
                self.continued_train_test_interval,
                self.continued_train_min_steps,
                self.continued_train_patience,
            ),
        }
        for profile_name, (total, interval, minimum, patience) in (
                training_profiles.items()):
            if min(total, interval, minimum, patience) <= 0:
                raise ValueError(
                    f"{profile_name} training profile values must be positive"
                )
            if interval > total:
                raise ValueError(
                    f"{profile_name} training test interval must not exceed "
                    "total steps"
                )
            if minimum > total:
                raise ValueError(
                    f"{profile_name} training minimum steps must not exceed "
                    "total steps"
                )
        if (self.num_warmup_steps < 0 or
                self.num_warmup_steps > min(
                    self.first_train_total_steps,
                    self.continued_train_total_steps,
                )):
            raise ValueError(
                "training warmup steps must be non-negative and no greater "
                "than either profile's total steps"
            )
        if self.training_min_free_mib < 0:
            raise ValueError("training_min_free_mib must not be negative")
        if not 0 <= self.training_max_gpu_utilization <= 100:
            raise ValueError(
                "training_max_gpu_utilization must be between 0 and 100"
            )
        if not 1 <= self.training_gpu_probe_samples <= 3:
            raise ValueError(
                "training_gpu_probe_samples must be between 1 and 3"
            )
        if not 0 <= self.training_gpu_probe_interval_seconds <= 1:
            raise ValueError(
                "training_gpu_probe_interval_seconds must be between 0 and 1"
            )
        if (not math.isfinite(self.training_waiter_lease_seconds) or
                self.training_waiter_lease_seconds < 15):
            raise ValueError(
                "training_waiter_lease_seconds must be at least 15"
            )
        if self.tokenizer_rayon_threads <= 0:
            raise ValueError("tokenizer_rayon_threads must be positive")
        if self.token_cache_entries < 0:
            raise ValueError("token_cache_entries must not be negative")
        if self.ts_serving_batch_size <= 0:
            raise ValueError("ts_serving_batch_size must be positive")
        if not 0 <= self.ts_max_batch_delay_ms <= 1000:
            raise ValueError(
                "ts_max_batch_delay_ms must be between 0 and 1000"
            )
        if self.report_paths is None:
            # Report paths are supplied by each registered task.
            self.report_paths = {}
        if self.target_funcs is None:
            self.target_funcs = {
                "kernel BUG in validate_xmit_skb": "validate_xmit_skb",
                "WARNING in tracepoint_add_func": "tracepoint_probe_register",
                "WARNING in cfg80211_connect": "cfg80211_connect",
                "KASAN: use-after-free Read in rxrpc_lookup_local": "rxrpc_lookup_local",
                "WARNING: suspicious RCU usage in kvm_vcpu_memslots": "kvm_vcpu_memslots",
            }
        if self.kallgraph_dirs is None:
            # Graph output paths are supplied per task when available.
            self.kallgraph_dirs = {}

    # ---- Inference backend ----
    inference_backend: str = "torchserve"                  # "torchserve" | "internal"

    # TorchServe specifics
    ts_host: str = "http://localhost"
    ts_inference_port: int = 37033  # gRPC inference port (not HTTP 37030)
    ts_management_port: int = 37031
    ts_model_name: str = "reach_filter"
    ts_handler_path: str = str(BRAIN_DIR / "handler.py")  # absolute path to TorchServe handler
    ts_model_store: str = str(BRAIN_DIR / "model_store")  # where .mar lives
    # Online scheduling is latency-sensitive. A 100 ms TorchServe batching
    # window dominated measured batch-1 latency, while concurrent fuzzer procs
    # already provide natural opportunities for small batches.
    ts_serving_batch_size: int = int(os.getenv(
        "SYZPILOT_TS_SERVING_BATCH_SIZE", "16"
    ))
    ts_max_batch_delay_ms: int = int(os.getenv(
        "SYZPILOT_TS_MAX_BATCH_DELAY_MS", "10"
    ))

    # if you manage TorchServe with your existing Flask manager, keep using it
    use_existing_ts_manager: bool = True                   # if True we call your operators directly


    # ---- Sharding & IO ----
    shard_size: int = 5000                                 # write one PKL every N samples to avoid RAM growth
    flush_interval_sec: int = 15

    # ---- Label interpretation ----
    # Fuzzer sends an exactly-one-hot vector whose index is already the class:
    # 0=Unreachable and 1..N=deepest reached waypoint.
    def bools_to_index(self, bools) -> int:
        selected = [index for index, value in enumerate(bools) if bool(value)]
        if len(selected) != 1:
            raise ValueError(f"label must be exactly one-hot: {list(bools)}")
        return selected[0]
