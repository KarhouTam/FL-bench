import os
import random
from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import numpy as np
from path import Path
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()
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


def clone_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )


def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module], requires_name=False
) -> Union[List[torch.Tensor], Tuple[List[str], List[torch.Tensor]]]:
    parameters = []
    keys = []
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(param)
                keys.append(name)

    if requires_name:
        return keys, parameters
    else:
        return parameters


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


class FLBenchOptimizer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.epoch_count = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        # self.epoch_count += 1
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if not self.scheduler else self.scheduler.state_dict(),
            # "epoch_count": self.epoch_count,
        }

    def load_state_dict(self, state_dict):
        # self.epoch_count = state_dict["epoch_count"]
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.scheduler:
            self.scheduler.load_state_dict(state_dict["scheduler"])
