import os
import random
from copy import deepcopy
from collections import Counter, OrderedDict
from typing import List, Tuple, Union
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from rich.console import Console

from data.utils.datasets import BaseDataset

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
OUT_DIR = PROJECT_DIR / "out"
TEMP_DIR = PROJECT_DIR / "temp"


def fix_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module],
    requires_name=False,
    detach=False,
) -> Union[List[torch.Tensor], Tuple[List[str], List[torch.Tensor]]]:
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)

    if requires_name:
        return keys, parameters
    else:
        return parameters


def vectorize(src, detach=True) -> torch.Tensor:
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    if isinstance(src, torch.nn.Module):
        return torch.cat([func(param).flatten() for param in trainable_params(src)])
    elif isinstance(src, list):
        return torch.cat([func(param).flatten() for param in src])
    elif isinstance(src, OrderedDict):
        return torch.cat([func(param).flatten() for param in src.values()])


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
) -> Tuple[float, float, int]:
    model.eval()
    correct = 0
    loss = 0
    sample_num = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        correct += (pred == y).sum().item()
        sample_num += len(y)
    return loss, correct, sample_num


def count_labels(dataset: BaseDataset, indices: List[int], min_value=0) -> List[int]:
    counter = Counter(dataset.targets[indices].tolist())
    return [counter.get(i, min_value) for i in range(len(dataset.classes))]


class Logger:
    def __init__(self, stdout: Console, enable_log, logfile_path):
        self.stdout = stdout
        self.logfile_stream = None
        self.enable_log = enable_log
        if self.enable_log:
            self.logfile_stream = open(logfile_path, "w")
            self.logger = Console(
                file=self.logfile_stream, record=True, log_path=False, log_time=False
            )

    def log(self, *args, **kwargs):
        self.stdout.log(*args, **kwargs)
        if self.enable_log:
            self.logger.log(*args, **kwargs)

    def close(self):
        if self.logfile_stream:
            self.logfile_stream.close()
