import json
import os
import random
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterator, Sequence, Union

import numpy as np
import pynvml
import torch
from omegaconf import DictConfig
from rich.console import Console
from torch.utils.data import DataLoader

from src.utils.constants import DEFAULT_COMMON_ARGS, DEFAULT_PARALLEL_ARGS
from src.utils.metrics import Metrics


def fix_random_seed(seed: int, use_cuda=False) -> None:
    """Fix the random seed of FL training.

    Args:
        seed (int): Any number you like as the random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available() and use_cuda:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimal_cuda_device(use_cuda: bool) -> torch.device:
    """Dynamically select CUDA device (has the most memory) for running FL
    experiment.

    Args:
        use_cuda (bool): `True` for using CUDA; `False` for using CPU only.

    Returns:
        torch.device: The selected CUDA device.
    """
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


def vectorize(
    src: OrderedDict[str, torch.Tensor] | list[torch.Tensor] | torch.nn.Module,
    detach=True,
) -> torch.Tensor:
    """Vectorize(Flatten) and concatenate all tensors in `src`.

    Args:
        `src`: The source of tensors.
        `detach`: Set as `True` to return `tensor.detach().clone()`. Defaults to `True`.

    Returns:
        The vectorized tensor.
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    if isinstance(src, list):
        return torch.cat([func(param).flatten() for param in src])
    elif isinstance(src, OrderedDict) or isinstance(src, dict):
        return torch.cat([func(param).flatten() for param in src.values()])
    elif isinstance(src, torch.nn.Module):
        return torch.cat([func(param).flatten() for param in src.state_dict().values()])
    elif isinstance(src, Iterator):
        return torch.cat([func(param).flatten() for param in src])


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
    model.to(device)
    metrics = Metrics()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y).item()
        pred = torch.argmax(logits, -1)
        metrics.update(Metrics(loss, pred, y))
    return metrics


def parse_args(
    config: DictConfig,
    method_name: str,
    get_method_args_func: Callable[[Sequence[str] | None], Namespace] | None,
) -> Namespace:
    """Purge arguments from default args dict, config file and CLI and produce
    the final arguments.

    Args:
        config_file_args (Union[dict, None]): Argument dictionary loaded from user-defined `.yml` file. `None` for unspecifying.
        method_name (str): The FL method's name.
        get_method_args_func (Union[ Callable[[Union[Sequence[str], None]], Namespace], None ]): The callable function of parsing FL method `method_name`'s spec arguments.
        method_args_list (list[str]): FL method `method_name`'s specified arguments set on CLI.

    Returns:
        DictConfig: The final argument namespace.
    """
    ARGS = dict(
        mode="serial", common=DEFAULT_COMMON_ARGS, parallel=DEFAULT_PARALLEL_ARGS
    )
    if "common" in config.keys():
        ARGS["common"].update(config["common"])
    if "parallel" in config.keys():
        ARGS["parallel"].update(config["parallel"])
    if "mode" in config.keys():
        ARGS["mode"] = config["mode"]
    if get_method_args_func is not None:
        ARGS[method_name] = get_method_args_func([]).__dict__

    for field in ["common", "parallel", method_name]:
        if field in config.keys():
            for key in config[field].keys():
                ARGS[field][key] = config[field][key]

    assert ARGS["mode"] in ["serial", "parallel"], f"Unrecongnized mode: {ARGS['mode']}"
    if ARGS["mode"] == "parallel":
        if ARGS["parallel"]["num_workers"] < 2:
            print(
                f"num_workers is less than 2: {ARGS['parallel']['num_workers']}, "
                "mode is fallback to serial."
            )
            ARGS["mode"] = "serial"
            del ARGS["parallel"]
    return DictConfig(ARGS)


class Logger:
    def __init__(
        self, stdout: Console, enable_log: bool, logfile_path: Union[Path, str]
    ):
        """This class is for solving the incompatibility between the progress
        bar and log function in library `rich`.

        Args:
            stdout (Console): The `rich.console.Console` for printing info onto stdout.
            enable_log (bool): Flag indicates whether log function is actived.
            logfile_path (Union[Path, str]): The path of log file.
        """
        self.stdout = stdout
        self.logfile_output_stream = None
        self.enable_log = enable_log
        if self.enable_log:
            self.logfile_output_stream = open(logfile_path, "w")
            self.logfile_logger = Console(
                file=self.logfile_output_stream,
                record=True,
                log_path=False,
                log_time=False,
                soft_wrap=True,
                tab_size=4,
            )

    def log(self, *args, **kwargs):
        self.stdout.log(*args, **kwargs)
        if self.enable_log:
            self.logfile_logger.log(*args, **kwargs)

    def close(self):
        if self.logfile_output_stream:
            self.logfile_output_stream.close()
