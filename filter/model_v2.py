
import torch
from torch import nn
from transformers import AutoModel

try:
    from common.curriculum import VALID_CURRICULUM_STAGES
except ImportError:  # Script execution keeps filter/ as the import root.
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from common.curriculum import VALID_CURRICULUM_STAGES

try:
    from .config import TOKEN
except ImportError:  # Script execution keeps filter/ as the import root.
    from config import TOKEN
try:
    from .utils import mean_pooling
except ImportError:  # Script execution keeps filter/ as the import root.
    from utils import mean_pooling


class TraceClassifierV2(nn.Module):
    def __init__(self, base_model: str, num_labels: int, stage: int=1):
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
            logits = curriculum_logits(logits, self.stage)
            labels = curriculum_labels(labels, self.num_classes, self.stage)
            output_labels = torch.argmax(logits, dim=1)
            loss = self.loss_fn(logits, labels)
            return loss, logits, output_labels, labels
        else:
            raise ValueError(f"Invalid output arg: `{output}`")


class TraceClassifierV2_2(TraceClassifierV2):
    def __init__(self, base_model, num_labels, stage = 1):
        super().__init__(base_model, num_labels, stage)
        self.classifier = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, self.base_model.config.intermediate_size),
            nn.ReLU(),
            nn.Linear(self.base_model.config.intermediate_size, num_labels),
        )


class TraceClassifierServingWrapper(nn.Module):
    """Expose logits with the exact objective semantics used during training."""

    def __init__(self, model: TraceClassifierV2, stage: int):
        super().__init__()
        if stage not in VALID_CURRICULUM_STAGES:
            raise ValueError(f"invalid training stage: {stage}")
        self.model = model
        self.stage = stage

    def forward(self, input_ids, attention_mask=None):
        logits = self.model(input_ids, attention_mask)
        return curriculum_logits(logits, self.stage)


def curriculum_logits(logits: torch.Tensor, stage: int) -> torch.Tensor:
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
    return logits


def curriculum_labels(
    labels: torch.Tensor, num_classes: int, stage: int
) -> torch.Tensor:
    """Map waypoint-level class indices to the active curriculum classes."""
    if stage not in VALID_CURRICULUM_STAGES:
        raise ValueError(f"invalid training stage: {stage}")
    if stage == 1:
        return (labels > 0).long()
    return labels
