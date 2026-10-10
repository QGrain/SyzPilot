
import torch
from torch import nn
from transformers import AutoModel

try:
    from common.curriculum import (
        VALID_CURRICULUM_STAGES,
        normalize_stage3_active_classes,
        stage2_deep_class_start,
    )
except ImportError:  # Script execution keeps filter/ as the import root.
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from common.curriculum import (
        VALID_CURRICULUM_STAGES,
        normalize_stage3_active_classes,
        stage2_deep_class_start,
    )

try:
    from .config import TOKEN
except ImportError:  # Script execution keeps filter/ as the import root.
    from config import TOKEN
try:
    from .utils import mean_pooling
except ImportError:  # Script execution keeps filter/ as the import root.
    from utils import mean_pooling


class TraceClassifierV2(nn.Module):
    def __init__(self, base_model: str, num_labels: int, stage: int = 1,
                 active_classes=None):
        super().__init__()
        if stage not in VALID_CURRICULUM_STAGES:
            raise ValueError(f"invalid training stage: {stage}")
        self.base_model = AutoModel.from_pretrained(base_model, token=TOKEN)
        self.num_classes = num_labels
        self.classifier = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, num_labels),
            # nn.Sigmoid(),
        )
        # self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = nn.CrossEntropyLoss()

        self.stage = stage
        self.active_classes = (
            normalize_stage3_active_classes(active_classes, num_labels)
            if stage == 3 else None
        )

    def forward(self, input_ids, attention_mask=None, labels=None, output="logits"):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        embedding = mean_pooling(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(embedding)

        if output == "logits":
            return logits
        elif output == "inference":
            pred_labels = torch.argmax(logits, dim=1)
            return logits, pred_labels
        elif output == "train":
            labels = torch.argmax(labels, dim=1)
            logits = curriculum_logits(
                logits, self.stage, self.active_classes
            )
            labels = curriculum_labels(
                labels, self.num_classes, self.stage, self.active_classes
            )
            output_labels = torch.argmax(logits, dim=1)
            loss = self.loss_fn(logits, labels)
            return loss, logits, output_labels, labels
        else:
            raise ValueError(f"Invalid output arg: `{output}`")


class TraceClassifierServingWrapper(nn.Module):
    """Expose logits with the exact objective semantics used during training."""

    def __init__(self, model: TraceClassifierV2, stage: int,
                 active_classes=None):
        super().__init__()
        if stage not in VALID_CURRICULUM_STAGES:
            raise ValueError(f"invalid training stage: {stage}")
        self.model = model
        self.stage = stage
        self.num_classes = int(model.num_classes)
        self.stage2_deep_start = (
            stage2_deep_class_start(self.num_classes) if stage == 2 else 0
        )
        model_active = getattr(model, "active_classes", None)
        if stage == 3:
            selected = normalize_stage3_active_classes(
                active_classes if active_classes is not None else model_active,
                self.num_classes,
            )
            if (model_active is not None and
                    normalize_stage3_active_classes(
                        model_active, self.num_classes
                    ) != selected):
                raise ValueError(
                    "serving active classes differ from the trained model"
                )
        else:
            selected = ()
        self.active_classes = selected or None
        active_indices = torch.tensor(selected, dtype=torch.long)
        inactive_indices = torch.tensor(
            tuple(
                index for index in range(self.num_classes)
                if index not in set(selected)
            ) if stage == 3 else (),
            dtype=torch.long,
        )
        self.register_buffer("stage3_active_indices", active_indices)
        self.register_buffer("stage3_inactive_indices", inactive_indices)
        self.stage3_has_other = bool(inactive_indices.numel())

    def forward(self, input_ids, attention_mask=None):
        logits = self.model(input_ids, attention_mask)
        if self.stage == 1:
            return torch.stack(
                [logits[:, 0], logits[:, 1:].mean(dim=1)], dim=1
            )
        if self.stage == 2:
            return torch.stack(
                [
                    logits[:, 0],
                    logits[:, 1:self.stage2_deep_start].mean(dim=1),
                    logits[:, self.stage2_deep_start:].mean(dim=1),
                ],
                dim=1,
            )
        grouped = logits.index_select(1, self.stage3_active_indices)
        if not self.stage3_has_other:
            return grouped
        other = torch.max(
            logits.index_select(1, self.stage3_inactive_indices), dim=1
        ).values
        return torch.cat((grouped, other.unsqueeze(1)), dim=1)


def curriculum_logits(logits: torch.Tensor, stage: int,
                      active_classes=None) -> torch.Tensor:
    """Group fixed-head logits according to the active curriculum objective."""
    if stage not in VALID_CURRICULUM_STAGES:
        raise ValueError(f"invalid training stage: {stage}")
    if (not torch.jit.is_tracing() and
            (logits.ndim != 2 or logits.shape[1] < 2)):
        raise ValueError("logits must have shape [batch, at least 2 classes]")
    if stage == 1:
        return torch.stack(
            [logits[:, 0], logits[:, 1:].mean(dim=1)], dim=1
        )
    if stage == 2:
        deep_start = stage2_deep_class_start(logits.shape[1])
        return torch.stack(
            [
                logits[:, 0],
                logits[:, 1:deep_start].mean(dim=1),
                logits[:, deep_start:].mean(dim=1),
            ],
            dim=1,
        )
    active = normalize_stage3_active_classes(
        active_classes, logits.shape[1]
    )
    indices = torch.tensor(active, device=logits.device)
    grouped = logits.index_select(1, indices)
    if len(active) == logits.shape[1]:
        return grouped
    active_lookup = set(active)
    inactive = tuple(
        index for index in range(logits.shape[1])
        if index not in active_lookup
    )
    inactive_indices = torch.tensor(inactive, device=logits.device)
    other = torch.max(
        logits.index_select(1, inactive_indices), dim=1
    ).values
    return torch.cat((grouped, other.unsqueeze(1)), dim=1)


def full_width_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    """Collapse a full exact-label response using rfilter's max decision."""
    if (not torch.jit.is_tracing() and
            (logits.ndim != 2 or logits.shape[1] < 2)):
        raise ValueError("logits must have shape [batch, at least 2 classes]")
    return torch.stack((
        logits[:, 0], torch.max(logits[:, 1:], dim=1).values
    ), dim=1)


def curriculum_labels(labels: torch.Tensor, num_classes: int, stage: int,
                      active_classes=None) -> torch.Tensor:
    """Map waypoint-level class indices to the active curriculum classes."""
    if stage not in VALID_CURRICULUM_STAGES:
        raise ValueError(f"invalid training stage: {stage}")
    if stage == 1:
        return (labels > 0).long()
    if stage == 2:
        deep_start = stage2_deep_class_start(num_classes)
        mapped = torch.zeros_like(labels, dtype=torch.long)
        mapped[(labels > 0) & (labels < deep_start)] = 1
        mapped[labels >= deep_start] = 2
        return mapped
    active = normalize_stage3_active_classes(active_classes, num_classes)
    lookup = torch.full(
        (num_classes,), len(active), dtype=torch.long, device=labels.device
    )
    lookup[list(active)] = torch.arange(
        len(active), dtype=torch.long, device=labels.device
    )
    return lookup[labels]
