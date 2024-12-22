import numpy as np
import torch
from sklearn import metrics


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


class Metrics:
    def __init__(self, loss=None, predicts=None, targets=None):
        self._loss = loss if loss is not None else 0.0
        self._targets = targets if targets is not None else []
        self._predicts = predicts if predicts is not None else []

    def update(self, other):
        if other is not None:
            self._predicts.extend(to_numpy(other._predicts))
            self._targets.extend(to_numpy(other._targets))
            self._loss += other._loss

    def _calculate(self, metric, **kwargs):
        return metric(self._targets, self._predicts, **kwargs)

    @property
    def loss(self):
        if len(self._targets) > 0:
            return self._loss / len(self._targets)
        else:
            return 0.0

    @property
    def macro_precision(self):
        score = self._calculate(
            metrics.precision_score, average="macro", zero_division=0
        )
        return score * 100

    @property
    def macro_recall(self):
        score = self._calculate(metrics.recall_score, average="macro", zero_division=0)
        return score * 100

    @property
    def micro_precision(self):
        score = self._calculate(
            metrics.precision_score, average="micro", zero_division=0
        )
        return score * 100

    @property
    def micro_recall(self):
        score = self._calculate(metrics.recall_score, average="micro", zero_division=0)
        return score * 100

    @property
    def accuracy(self):
        if self.size == 0:
            return 0
        score = self._calculate(metrics.accuracy_score)
        return score * 100

    @property
    def corrects(self):
        return self._calculate(metrics.accuracy_score, normalize=False)

    @property
    def size(self):
        return len(self._targets)

    def __bool__(self):
        return len(self._targets) > 0 and len(self._predicts) > 0
