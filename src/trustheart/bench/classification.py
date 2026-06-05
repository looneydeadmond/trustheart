from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def _accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    return float((preds == targets).mean())


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    preds = probs.argmax(axis=-1)
    y = targets.detach().cpu().numpy()
    present = np.unique(y)
    metrics: dict[str, float] = {
        "accuracy": _accuracy(preds, y),
        "precision": float(precision_score(y, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(y, preds, average="macro", zero_division=0)),
        "f1": float(f1_score(y, preds, average="macro", zero_division=0)),
    }
    if present.shape[0] > 1 and probs.shape[1] > 1:
        try:
            metrics["auroc"] = float(
                roc_auc_score(
                    y, probs, multi_class="ovr", average="macro", labels=np.arange(probs.shape[1])
                )
            )
        except ValueError:
            metrics["auroc"] = float("nan")
    else:
        metrics["auroc"] = float("nan")
    return metrics
