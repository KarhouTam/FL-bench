import json
import os
import random
import yaml
from argparse import Namespace
from collections import Counter, OrderedDict
from typing import Callable, Dict, List, Sequence, Tuple, Union
from pathlib import Path

import torch
import pynvml
import numpy as np
from torch.utils.data import DataLoader
from rich.console import Console

from data.utils.datasets import BaseDataset
from src.utils.metrics import Metrics
from src.utils.constants import DEFAULT_COMMON_ARGS


def fix_random_seed(seed: int) -> None:
    """Fix the random seed of FL training.

    Args:
        seed (int): Any number you like as the random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimal_cuda_device(use_cuda: bool) -> torch.device:
    """Dynamically select CUDA device (has the most memory) for running FL experiment.

    Args:
        use_cuda (bool): `True` for using CUDA; `False` for using CPU only.

    Returns:
        torch.device: The selected CUDA device.
    """
    # This function is modified by the `get_best_gpu()` in https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/functional.py
    # Shout out to FedLab, which is an incredible FL framework!
    if not torch.cuda.is_available() or not use_cuda:
        return torch.device("cpu")
    pynvml.nvmlInit()
    gpu_memory = []
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_ids = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        assert max(gpu_ids) < torch.cuda.device_count()
    else:
        gpu_ids = range(torch.cuda.device_count())

    for i in gpu_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory.append(memory_info.free)
    gpu_memory = np.array(gpu_memory)
    best_gpu_id = np.argmax(gpu_memory)
    return torch.device(f"cuda:{best_gpu_id}")


def trainable_params(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module],
    detach=False,
    requires_name=False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]:
    """Collect all parameters in `src` that `.requires_grad = True` into a list and return it.

    Args:
        src (Union[OrderedDict[str, torch.Tensor], torch.nn.Module]): The source that contains parameters.
        requires_name (bool, optional): If set to `True`, The names of parameters would also return in another list. Defaults to False.
        detach (bool, optional): If set to `True`, the list would contain `param.detach().clone()` rather than `param`. Defaults to False.

    Returns:
        List of parameters [, names of parameters].
    """
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
        return parameters, keys
    else:
        return parameters


def vectorize(
    src: Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], detach=True
) -> torch.Tensor:
    """Vectorize and concatenate all tensors in `src`.

    Args:
        src (Union[OrderedDict[str, torch.Tensor]List[torch.Tensor]]): The source of tensors.
        detach (bool, optional): Set to `True`, return the `.detach().clone()`. Defaults to True.

    Returns:
        torch.Tensor: The vectorized tensor.
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    if isinstance(src, list):
        return torch.cat([func(param).flatten() for param in src])
    elif isinstance(src, OrderedDict):
        return torch.cat([func(param).flatten() for param in src.values()])


@torch.no_grad()
def evalutate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
) -> Metrics:
    """For evaluating the `model` over `dataloader` and return metrics.

    Args:
        model (torch.nn.Module): Target model.
        dataloader (DataLoader): Target dataloader.
        criterion (optional): The metric criterion. Defaults to torch.nn.CrossEntropyLoss(reduction="sum").
        device (torch.device, optional): The device that holds the computation. Defaults to torch.device("cpu").

    Returns:
        Metrics: The metrics objective.
    """
    model.eval()
    metrics = Metrics()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        metrics.update(Metrics(loss, pred, y))
    return metrics


def count_labels(
    dataset: BaseDataset, indices: List[int] = None, min_value=0
) -> List[int]:
    """For counting number of labels in `dataset.targets`.

    Args:
        dataset (BaseDataset): Target dataset.
        indices (List[int]): the subset indices. Defaults to all indices of `dataset` if not specified.
        min_value (int, optional): The minimum value for each label. Defaults to 0.

    Returns:
        List[int]: The number of each label.
    """
    if indices is None:
        indices = list(range(len(dataset.targets)))
    counter = Counter(dataset.targets[indices].tolist())
    return [counter.get(i, min_value) for i in range(len(dataset.classes))]


def parse_args(
    config_file_path: str,
    method_name: str,
    get_method_args_func: Union[
        Callable[[Union[Sequence[str], None]], Namespace], None
    ],
    method_args_list: List[str],
) -> Namespace:
    """Merging default argument namespace with argument dict from custom config file.

    Args:
        default_args (Namespace): Default args set by CLI.

    Returns:
        Namespace: The merged arg namespace.
    """
    final_args = NestedNamespace({"common": DEFAULT_COMMON_ARGS})
    config_file_args = {}
    if config_file_path is not None:
        try:
            with open(Path(config_file_path).absolute()) as f:
                config_file_args = yaml.safe_load(f)
            final_args = NestedNamespace(config_file_args)

            common_args = DEFAULT_COMMON_ARGS
            common_args.update(config_file_args["common"])
            final_args.__setattr__("common", NestedNamespace(common_args))
        except:
            Warning(
                f"Unrecongnized config file path: {Path(config_file_path).absolute()}. All common arguments are rolled back to their defaults."
            )

    if get_method_args_func is not None:
        default_method_args = get_method_args_func([]).__dict__
        config_file_method_args = config_file_args.get(method_name, {})
        cli_method_args = get_method_args_func(method_args_list).__dict__

        # extract arguments set explicitly set in CLI
        for key in default_method_args.keys():
            if default_method_args[key] == cli_method_args[key]:
                cli_method_args.pop(key)

        # For the same argument, the value setting priority is CLI > config file > defalut value
        method_args = default_method_args
        for key in default_method_args.keys():
            if key in cli_method_args.keys():
                method_args[key] = cli_method_args[key]
            elif key in config_file_method_args.keys():
                method_args[key] = config_file_method_args[key]

        final_args.__setattr__(method_name, NestedNamespace(method_args))

    return final_args


class Logger:
    def __init__(
        self, stdout: Console, enable_log: bool, logfile_path: Union[Path, str]
    ):
        """This class is for solving the incompatibility between the progress bar and log function in library `rich`.

        Args:
            stdout (Console): The `rich.console.Console` for printing info onto stdout.
            enable_log (bool): Flag indicates whether log function is actived.
            logfile_path (Union[Path, str]): The path of log file.
        """
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


class NestedNamespace(Namespace):
    def __init__(self, args_dict: Dict):
        super().__init__(
            **{
                key: self._nested_namespace(value) if isinstance(value, dict) else value
                for key, value in args_dict.items()
            }
        )

    def _nested_namespace(self, dictionary):
        return NestedNamespace(dictionary)

    def to_dict(self):
        return {
            key: (value.to_dict() if isinstance(value, NestedNamespace) else value)
            for key, value in self.__dict__.items()
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4, sort_keys=False)
