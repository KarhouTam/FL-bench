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
from torch.utils.data import DataLoader, Subset

from src.utils.constants import DEFAULTS
from src.utils.metrics import Metrics


def fix_random_seed(seed: int, use_cuda=False) -> None:
    """Fix the random seed of FL training.

    Args:
        seed: Any number you like as the random seed.
        use_cuda: Flag indicates if using cuda.
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
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
    model_in_train_mode: bool = False,
) -> Metrics:
    """For evaluating the `model` over `dataloader` and return metrics.

    Args:
        model (torch.nn.Module): Target model.
        dataloader (DataLoader): Target dataloader.
        criterion (optional): The metric criterion. Defaults to torch.nn.CrossEntropyLoss(reduction="sum").
        device (torch.device, optional): The device that holds the computation. Defaults to torch.device("cpu").
        model_in_eval_mode (bool, optional): Set as `True` to switch model to eval mode. Defaults to `True`.

    Returns:
        Metrics: The metrics objective.
    """
    if model_in_train_mode:
        model.train()
    else:
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
) -> DictConfig:
    """Purge arguments from default args dict, config file and CLI and produce
    the final arguments.

    Args:
        config: DictConfig set from .yaml config file.
        method_name: The FL method's name.
        get_method_args_func: The callable function of parsing FL method `method_name`'s spec arguments.
    Returns:
        DictConfig: The final argument namespace.
    """
    final_args = DictConfig(DEFAULTS)

    def _merge_configs(defaults: DictConfig, config: DictConfig) -> DictConfig:
        merged = DictConfig({})
        for key, default_value in defaults.items():
            if key in config:
                if isinstance(default_value, DictConfig) and isinstance(
                    config[key], DictConfig
                ):
                    merged[key] = _merge_configs(default_value, config[key])
                else:
                    merged[key] = config[key]
            else:
                merged[key] = default_value
        return merged

    final_args = _merge_configs(final_args, config)

    if hasattr(config, method_name):
        final_args[method_name] = config[method_name]

    if get_method_args_func is not None:
        default_method_args = DictConfig(get_method_args_func([]).__dict__)
        if hasattr(final_args, method_name):
            for key in default_method_args.keys():
                if key not in final_args[method_name].keys():
                    final_args[method_name][key] = default_method_args[key]
        else:
            final_args[method_name] = default_method_args

    assert final_args.mode in [
        "serial",
        "parallel",
    ], f"Unrecongnized mode: {final_args.mode}"
    if final_args.mode == "parallel":
        import ray

        num_available_gpus = final_args.parallel.num_gpus
        num_available_cpus = final_args.parallel.num_cpus
        if num_available_gpus is None:
            pynvml.nvmlInit()
            num_total_gpus = pynvml.nvmlDeviceGetCount()
            if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
                num_available_gpus = min(
                    len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")), num_total_gpus
                )
            else:
                num_available_gpus = num_total_gpus
        if num_available_cpus is None:
            num_available_cpus = os.cpu_count()

        try:
            ray.init(
                address=config.parallel.ray_cluster_addr,
                namespace=method_name,
                num_cpus=num_available_cpus,
                num_gpus=num_available_gpus,
                ignore_reinit_error=True,
            )
        except ValueError:
            # have existing cluster
            # then ignore num_cpus and num_gpus
            ray.init(
                address=config.parallel.ray_cluster_addr,
                namespace=method_name,
                ignore_reinit_error=True,
            )

        cluster_resources = ray.cluster_resources()
        final_args.parallel.num_cpus = cluster_resources["CPU"]
        final_args.parallel.num_gpus = cluster_resources["GPU"]
        if final_args.parallel.num_workers < 2:
            print(
                f"num_workers is less than 2: {final_args.parallel.num_workers}, "
                "mode fallbacks to serial."
            )
            final_args.mode = "serial"
            del final_args.parallel

    return final_args


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


def initialize_data_loaders(
    dataset: torch.utils.data.Dataset,
    data_indices: list[dict[str, list[int]]],
    batch_size: int=32,
) -> tuple[DataLoader, DataLoader, DataLoader, Subset, Subset, Subset]:
    """
    Initializes and returns data loaders for training, validation, and testing.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be used for loading data.
        data_indices (list[dict[str, list[int]]]): A list of dictionaries containing indices for train, validation, and test data.
        batch_size (int, optional): The batch size for the data loaders. Defaults to 32.

    Returns:
        tuple: A tuple containing the following elements:
            - trainloader (torch.utils.data.DataLoader): The data loader for training data.
            - valloader (torch.utils.data.DataLoader): The data loader for validation data.
            - testloader (torch.utils.data.DataLoader): The data loader for testing data.
            - trainset (torch.utils.data.Subset): The subset of the dataset containing training data.
            - valset (torch.utils.data.Subset): The subset of the dataset containing validation data.
            - testset (torch.utils.data.Subset): The subset of the dataset containing testing data.
    """
    val_indices = np.concatenate(
        [client_i_indices["val"] for client_i_indices in data_indices]
    )
    test_indices = np.concatenate(
        [client_i_indices["test"] for client_i_indices in data_indices]
    )
    train_indices = np.concatenate(
        [client_i_indices["train"] for client_i_indices in data_indices]
    )
    valset = torch.utils.data.Subset(dataset, val_indices)
    testset = torch.utils.data.Subset(dataset, test_indices)
    trainset = torch.utils.data.Subset(dataset, train_indices)
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False,
    )

    return trainloader, testloader, valloader, trainset, testset, valset
