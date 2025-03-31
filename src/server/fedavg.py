import functools
import inspect
import json
import os
import pickle
import random
import shutil
import time
import traceback
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import ray
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.pretty import pprint as rich_pprint
from rich.progress import track
from torchvision import transforms

from data.utils.datasets import DATASETS, BaseDataset
from src.client.fedavg import FedAvgClient
from src.utils.constants import (
    DATA_MEAN,
    DATA_STD,
    FLBENCH_ROOT,
    LR_SCHEDULERS,
    MODE,
    OPTIMIZERS,
)
from src.utils.functional import (
    evaluate_model,
    fix_random_seed,
    get_optimal_cuda_device,
    initialize_data_loaders,
)
from src.utils.logger import Logger
from src.utils.metrics import Metrics
from src.utils.models import MODELS, DecoupledModel
from src.utils.trainer import FLbenchTrainer


class FedAvgServer:
    algorithm_name = "FedAvg"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedAvgClient

    def __init__(
        self,
        args: DictConfig,
        init_trainer=True,
        init_model=True,
    ):
        """
        Args:
            `args`: A DictConfig object of the arguments.
            `init_trainer`: `True` indicates that initializing trainer now (with no extra arguments); `False` for explicitly initializing afterwards.
            `init_model`: `True` indicates that initializing model parameters; `False` for explicitly initializing afterwards.
        """
        self.args = args

        self.device = get_optimal_cuda_device(self.args.common.use_cuda)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.set_device(self.device)

        fix_random_seed(self.args.common.seed, use_cuda=self.device.type == "cuda")

        self.output_dir = Path(HydraConfig.get().runtime.output_dir)
        with open(
            FLBENCH_ROOT / "data" / self.args.dataset.name / "args.json", "r"
        ) as f:
            self.args.dataset.update(DictConfig(json.load(f)))

        # get client party info
        try:
            partition_path = (
                FLBENCH_ROOT / "data" / self.args.dataset.name / "partition.pkl"
            )
            with open(partition_path, "rb") as f:
                self.data_partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {self.args.dataset.name} first.")
        self.train_clients: List[int] = self.data_partition["separation"]["train"]
        self.test_clients: List[int] = self.data_partition["separation"]["test"]
        self.val_clients: List[int] = self.data_partition["separation"]["val"]
        self.client_num: int = self.data_partition["separation"]["total"]

        # init model(s) parameters
        if init_model:
            self.init_model()

        self.client_optimizer_states = {i: {} for i in range(self.client_num)}

        self.client_lr_scheduler_states = {i: {} for i in range(self.client_num)}

        self.client_local_epoches: List[int] = [
            self.args.common.local_epoch
        ] * self.client_num

        # system heterogeneity (straggler) setting
        if (
            self.args.common.straggler_ratio > 0
            and self.args.common.local_epoch
            > self.args.common.straggler_min_local_epoch
        ):
            straggler_num = int(self.client_num * self.args.common.straggler_ratio)
            normal_num = self.client_num - straggler_num
            self.client_local_epoches = [self.args.common.local_epoch] * (
                normal_num
            ) + random.choices(
                range(
                    self.args.common.straggler_min_local_epoch,
                    self.args.common.local_epoch,
                ),
                k=straggler_num,
            )
            random.shuffle(self.client_local_epoches)

        # To make sure all algorithms run through the same client sampling stream.
        # Some algorithms' implicit operations at client side may
        # disturb the stream if sampling happens at each FL round's beginning.
        self.client_sample_stream = [
            random.sample(
                self.train_clients,
                max(1, int(self.client_num * self.args.common.join_ratio)),
            )
            for _ in range(self.args.common.global_epoch)
        ]
        self.selected_clients: List[int] = []
        self.current_epoch = 0

        # For controlling behaviors of some specific methods while testing (not used by all methods)
        self.testing = False

        if not os.path.isdir(self.output_dir) and (
            self.args.common.save_log
            or self.args.common.save_learning_curve_plot
            or self.args.common.save_metrics
        ):
            os.makedirs(self.output_dir, exist_ok=True)

        self.client_metrics = {i: {} for i in self.train_clients}
        self.aggregated_client_metrics = {
            "before": {"train": [], "val": [], "test": []},
            "after": {"train": [], "val": [], "test": []},
        }

        self.verbose = False
        stdout = Console(log_path=False, log_time=False, soft_wrap=True, tab_size=4)
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.common.save_log,
            logfile_path=self.output_dir / "main.log",
        )
        self.test_results: Dict[int, Dict[str, Dict[str, Metrics]]] = {}
        self.train_progress_bar = track(
            range(self.args.common.global_epoch),
            "[bold green]Training...",
            console=stdout,
        )

        if self.args.common.monitor is not None:
            self.monitor_window_name_suffix = (
                self.args.dataset.monitor_window_name_suffix
            )

        if self.args.common.monitor == "visdom":
            from visdom import Visdom

            self.viz = Visdom()
        elif self.args.common.monitor == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(log_dir=self.output_dir)

        # init dataset
        self.dataset = self.get_dataset()
        self.client_data_indices = self.get_clients_data_indices()

        # init trainer
        self.trainer: FLbenchTrainer = None
        if self.client_cls is None or not issubclass(self.client_cls, FedAvgClient):
            raise ValueError(f"{self.client_cls} is not a subclass of {FedAvgClient}.")
        if init_trainer:
            self.init_trainer()

        # create setup for centralized evaluation
        if 0 < self.args.common.test.server.interval <= self.args.common.global_epoch:
            if self.all_model_params_personalized:
                self.logger.warn(
                    "Warning: Centralized evaluation is not supported for unique model setting."
                )
            else:
                (
                    self.trainloader,
                    self.testloader,
                    self.valloader,
                    self.trainset,
                    self.testset,
                    self.valset,
                ) = initialize_data_loaders(
                    self.dataset, self.client_data_indices, self.args.common.batch_size
                )

    def init_model(
        self,
        model: Optional[DecoupledModel] = None,
        preprocess_func: Optional[Callable[[DecoupledModel], None]] = None,
        postprocess_func: Optional[Callable[[DecoupledModel], None]] = None,
    ):
        """Initialize the global model and client personal model parameters.

            model: The global model. If not provided, will use the default model specified in `args.model.name`. Defaults to None.
            preprocess_func: A function that takes the global model as input and preprocesses it. Defaults to None.
            postprocess_func: A function that takes the global model as input and postprocesses it. Defaults to None.

        Raises:
            FileNotFoundError: If `external_model_weights_path` is not a valid file path or not a `.pt` file.
            TypeError: If `external_model_weights_path` is not a valid `.pt` file.
        """
        if model is None:
            self.model: DecoupledModel = MODELS[self.args.model.name](
                dataset=self.args.dataset.name,
                pretrained=self.args.model.use_torchvision_pretrained_weights,
            )
        else:
            self.model = model
        self.model.check_and_preprocess(self.args)

        if preprocess_func is not None:
            preprocess_func(self.model)

        _init_global_params, _init_global_params_name = [], []
        for key, param in self.model.named_parameters():
            _init_global_params.append(param.data.clone())
            _init_global_params_name.append(key)

        self.public_model_param_names = _init_global_params_name
        self.public_model_params: OrderedDict[str, torch.Tensor] = OrderedDict(
            zip(_init_global_params_name, _init_global_params)
        )

        if self.args.model.external_model_weights_path is not None:
            file_path = str(
                (FLBENCH_ROOT / self.args.model.external_model_weights_path).absolute()
            )
            if os.path.isfile(file_path) and file_path.find(".pt") != -1:
                self.public_model_params.update(
                    torch.load(file_path, map_location="cpu")
                )
            elif not os.path.isfile(file_path):
                raise FileNotFoundError(f"{file_path} is not a valid file path.")
            elif file_path.find(".pt") == -1:
                raise TypeError(f"{file_path} is not a valid .pt file.")

        self.clients_personal_model_params = {i: {} for i in range(self.client_num)}

        if self.args.common.buffers == "local":
            _init_buffers = OrderedDict(self.model.named_buffers())
            for i in range(self.client_num):
                self.clients_personal_model_params[i] = deepcopy(_init_buffers)

        if self.all_model_params_personalized:
            for params_dict in self.clients_personal_model_params.values():
                params_dict.update(deepcopy(self.model.state_dict()))

        if postprocess_func is not None:
            postprocess_func(self.model)

    def init_trainer(self, **extras):
        """Initiate the FL-bench trainier that responsible to client training.

        Args:
            `extras`: Arguments of `self.client_cls.__init__()` that NOT included in
        `[model, args, optimizer_cls, lr_scheduler_cls, dataset, data_indices,
        device, return_diff]`.
        """
        if self.args.mode == "serial" or self.args.parallel.num_workers < 2:
            self.trainer = FLbenchTrainer(
                server=self,
                client_cls=self.client_cls,
                mode=MODE.SERIAL,
                num_workers=0,
                init_args=dict(
                    model=deepcopy(self.model),
                    optimizer_cls=self.get_client_optimizer_cls(),
                    lr_scheduler_cls=self.get_client_lr_scheduler_cls(),
                    args=self.args,
                    dataset=self.dataset,
                    data_indices=self.client_data_indices,
                    device=self.device,
                    return_diff=self.return_diff,
                    **extras,
                ),
            )
        else:
            model_ref = ray.put(self.model.cpu())
            optimzier_cls_ref = ray.put(self.get_client_optimizer_cls())
            lr_scheduler_cls_ref = ray.put(self.get_client_lr_scheduler_cls())
            dataset_ref = ray.put(self.dataset)
            data_indices_ref = ray.put(self.client_data_indices)
            args_ref = ray.put(self.args)
            device_ref = ray.put(None)  # in parallel mode, workers decide their device
            return_diff_ref = ray.put(self.return_diff)
            self.trainer = FLbenchTrainer(
                server=self,
                client_cls=self.client_cls,
                mode=MODE.PARALLEL,
                num_workers=int(self.args.parallel.num_workers),
                init_args=dict(
                    model=model_ref,
                    optimizer_cls=optimzier_cls_ref,
                    lr_scheduler_cls=lr_scheduler_cls_ref,
                    args=args_ref,
                    dataset=dataset_ref,
                    data_indices=data_indices_ref,
                    device=device_ref,
                    return_diff=return_diff_ref,
                    **{key: ray.put(value) for key, value in extras.items()},
                ),
            )

    def get_clients_data_indices(self) -> list[dict[str, list[int]]]:
        """Gets a list of client data indices.

        Load and return the client-side data index from the partition file for the specified dataset.

        Raises:
            FileNotFoundError: If the partition file does not exist.

        Returns:
        list[dict[str, list[int]]]: A list of client-side data indexes, where each element is a dictionary,
        Contains the keys "train", "val", and "test" for a list of data indexes for each partition.
        """
        try:
            partition_path = (
                FLBENCH_ROOT / "data" / self.args.dataset.name / "partition.pkl"
            )
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {self.args.dataset.name} first.")

        # [0: {"train": [...], "val": [...], "test": [...]}, ...]
        data_indices: list[dict[str, list[int]]] = partition["data_indices"]

        return data_indices

    def get_dataset(self) -> BaseDataset:
        """Load the specified dataset according to the configuration.

        Returns:
        BaseDataset: This is the loaded dataset instance,
        which inherits from the BaseDataset class.
        """
        dataset: BaseDataset = DATASETS[self.args.dataset.name](
            root=FLBENCH_ROOT / "data" / self.args.dataset.name,
            args=self.args.dataset,
            **self.get_dataset_transforms(),
        )

        return dataset

    def get_dataset_transforms(self):
        """Define data preprocessing schemes. These schemes will work for every
        client. Consider to overwrite this function for your unique data
        preprocessing.

        Returns:
            Dict[str, Callable], which includes keys:
                `train_data_transform`: The transform for training data.
                `train_target_transform`: The transform for training targets.
                `test_data_transform`: The transform for testing data.
                `test_target_transform`: The transform for testing targets.
        """
        test_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[self.args.dataset.name], DATA_STD[self.args.dataset.name]
                )
            ]
            if self.args.dataset.name in DATA_MEAN
            and self.args.dataset.name in DATA_STD
            else []
        )
        test_target_transform = transforms.Compose([])
        train_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[self.args.dataset.name], DATA_STD[self.args.dataset.name]
                )
            ]
            if self.args.dataset.name in DATA_MEAN
            and self.args.dataset.name in DATA_STD
            else []
        )
        train_target_transform = transforms.Compose([])
        return dict(
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
        )

    def get_client_optimizer_cls(self) -> type[torch.optim.Optimizer]:
        """Get client-side model training optimizer.

        Returns:
            A partial initiated optimizer class that client only need to add `params` arg.
        """
        target_optimizer_cls: type[torch.optim.Optimizer] = OPTIMIZERS[
            self.args.optimizer.name
        ]
        keys_required = inspect.getfullargspec(target_optimizer_cls.__init__).args
        args_valid = {}
        for key, value in self.args.optimizer.items():
            if key in keys_required:
                args_valid[key] = value

        optimizer_cls = functools.partial(target_optimizer_cls, **args_valid)
        args_valid["name"] = self.args.optimizer.name
        self.args.optimizer = DictConfig(args_valid)
        return optimizer_cls

    def get_client_lr_scheduler_cls(
        self,
    ) -> Union[type[torch.optim.lr_scheduler.LRScheduler], None]:
        """Get the client-side learning rate scheduler class. Return None if
        lr_scheduler.name is NOne or no lr_scheduler arguement is provided.

        Returns:
            None or a partial initiated lr_scheduler class that client only need to add `optimizer` arg.
        """
        if hasattr(self.args, "lr_scheduler"):
            if self.args.lr_scheduler.name is None:
                del self.args.lr_scheduler
                return None
            lr_scheduler_args = getattr(self.args, "lr_scheduler")
            if lr_scheduler_args.name is not None:
                target_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] = (
                    LR_SCHEDULERS[lr_scheduler_args.name]
                )
                keys_required = inspect.getfullargspec(
                    target_scheduler_cls.__init__
                ).args

                args_valid = {}
                for key, value in self.args.lr_scheduler.items():
                    if key in keys_required:
                        args_valid[key] = value

                lr_scheduler_cls = functools.partial(target_scheduler_cls, **args_valid)
                args_valid["name"] = self.args.lr_scheduler.name
                self.args.lr_scheduler = DictConfig(args_valid)
                return lr_scheduler_cls
        else:
            return None

    def train(self):
        avg_round_time = 0
        for E in self.train_progress_bar:
            self.current_epoch = E
            self.verbose = (self.current_epoch + 1) % self.args.common.verbose_gap == 0

            if self.verbose:
                self.logger.log("-" * 28, f"TRAINING EPOCH: {E + 1}", "-" * 28)

            self.selected_clients = self.client_sample_stream[E]
            begin = time.time()
            self.train_one_round()
            end = time.time()
            avg_round_time = (avg_round_time * self.current_epoch + (end - begin)) / (
                self.current_epoch + 1
            )

            if (
                self.args.common.test.server.interval > 0
                and (E + 1) % self.args.common.test.server.interval == 0
            ):
                self.test_global_model()
            if (
                self.args.common.test.client.interval > 0
                and (E + 1) % self.args.common.test.client.interval == 0
            ):
                self.test_client_models()

            self.display_metrics()

        self.logger.log(
            f"{self.algorithm_name}'s average time taken by each global epoch: "
            f"{int(avg_round_time // 60)} min {(avg_round_time % 60):.2f} sec."
        )

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at
        server side) in each communication round."""

        client_packages = self.trainer.train()
        self.aggregate_client_updates(client_packages)

    def package(self, client_id: int):
        """Package parameters that the client-side training needs. If you are
        implementing your own FL method and your method has different
        parameters to FedAvg's that passes from server-side to client-side,
        this method need to be overrided. All this method should do is
        returning a dict that contains all parameters.

        Args:
            client_id: The client ID.

        Returns:
            A dict of parameters: {
                `client_id`: The client ID.
                `local_epoch`: The num of epoches that client local training performs.
                `client_model_params`: The client model parameter dict.
                `optimizer_state`: The client model optimizer's state dict.
                `lr_scheduler_state`: The client learning scheduler's state dict.
                `return_diff`: Flag that indicates whether client should send parameters difference.
                    `False`: Client sends vanilla model parameters;
                    `True`: Client sends `diff = global - local`.
            }.
        """
        return dict(
            client_id=client_id,
            local_epoch=self.client_local_epoches[client_id],
            **self.get_client_model_params(client_id),
            optimizer_state=self.client_optimizer_states[client_id],
            lr_scheduler_state=self.client_lr_scheduler_states[client_id],
            return_diff=self.return_diff,
        )

    def test_client_models(self):
        """The function for testing FL method's output (a single global model
        or personalized client models)."""
        self.testing = True
        clients = list(set(self.val_clients + self.test_clients))
        template = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }
        if len(clients) > 0:
            if self.val_clients == self.train_clients == self.test_clients:
                results = {"all_clients": template}
                self.trainer.test(clients, results["all_clients"])
            else:
                results = {
                    "val_clients": deepcopy(template),
                    "test_clients": deepcopy(template),
                }
                if len(self.val_clients) > 0:
                    self.trainer.test(self.val_clients, results["val_clients"])
                if len(self.test_clients) > 0:
                    self.trainer.test(self.test_clients, results["test_clients"])

            if self.current_epoch + 1 not in self.test_results:
                self.test_results[self.current_epoch + 1] = results
            else:
                self.test_results[self.current_epoch + 1].update(results)

        self.testing = False

    def test_global_model(self):
        """The function for testing FL method's output (a single global model
        or personalized client models)."""
        # Has client personal model parameters, centralized evaluation for the global model is not available.
        if any(
            len(params_dict)
            for params_dict in self.clients_personal_model_params.values()
        ):
            return
        self.model.load_state_dict(self.public_model_params, strict=False)
        self.testing = True
        metrics = self.evaluate(
            model_in_train_mode=self.args.common.test.server.model_in_train_mode
        )

        if self.current_epoch + 1 not in self.test_results:
            self.test_results[self.current_epoch + 1] = {
                "centralized": {"before": metrics, "after": metrics}
            }
        else:
            self.test_results[self.current_epoch + 1]["centralized"] = {
                "before": metrics,
                "after": metrics,
            }

        self.testing = False

    @torch.no_grad()
    def evaluate(
        self, model: torch.nn.Module = None, model_in_train_mode: bool = True
    ) -> dict[str, Metrics]:
        """Evaluating server model.

        Args:
            model: Used model. Defaults to None, which will fallback to `self.model`.

        Returns:
            A evalution results dict: {
                `train`: results on client training set.
                `val`: results on client validation set.
                `test`: results on client test set.
            }
        """
        target_model = self.model if model is None else model
        self.dataset.eval()
        train_metrics = Metrics()
        val_metrics = Metrics()
        test_metrics = Metrics()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if len(self.testset) > 0 and self.args.common.test.server.test:
            test_metrics = evaluate_model(
                model=target_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
                model_in_train_mode=model_in_train_mode,
            )

        if len(self.valset) > 0 and self.args.common.test.server.val:
            val_metrics = evaluate_model(
                model=target_model,
                dataloader=self.valloader,
                criterion=criterion,
                device=self.device,
                model_in_train_mode=model_in_train_mode,
            )

        if len(self.trainset) > 0 and self.args.common.test.server.train:
            train_metrics = evaluate_model(
                model=target_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
                model_in_train_mode=model_in_train_mode,
            )
        return {"train": train_metrics, "val": val_metrics, "test": test_metrics}

    def get_client_model_params(self, client_id: int) -> OrderedDict[str, torch.Tensor]:
        """This function is for outputting model parameters that asked by
        `client_id`.

        Args:
            client_id (int): The ID of query client.

        Returns:
            {
                `regular_model_params`: Generally model parameters that join aggregation.
                `personal_model_params`: Client personal model parameters that won't join aggregation.
            }
        """
        regular_params = deepcopy(self.public_model_params)
        personal_params = self.clients_personal_model_params[client_id]
        return dict(
            regular_model_params=regular_params, personal_model_params=personal_params
        )

    @torch.no_grad()
    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        """Aggregate clients model parameters and produce global model
        parameters.

        Args:
            client_packages: Dict of client parameter packages, with format:
            {
                `client_id`: {
                    `regular_model_params`: ...,
                    `optimizer_state`: ...,
                }
            }

            About the content of client parameter package, check `FedAvgClient.package()`.
        """
        client_weights = [package["weight"] for package in client_packages.values()]
        weights = torch.tensor(client_weights) / sum(client_weights)
        if self.return_diff:  # inputs are model params diff
            for name, global_param in self.public_model_params.items():
                diffs = torch.stack(
                    [
                        package["model_params_diff"][name]
                        for package in client_packages.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(diffs * weights, dim=-1)
                self.public_model_params[name].data -= aggregated
        else:
            for name, global_param in self.public_model_params.items():
                client_params = torch.stack(
                    [
                        package["regular_model_params"][name]
                        for package in client_packages.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(client_params * weights, dim=-1)

                global_param.data = aggregated
        self.model.load_state_dict(self.public_model_params, strict=False)

    def display_metrics(self):
        """Display aggregated client and server evaluation metrics at each
        round.

        This method aggregates metrics from selected clients for both
        'before' and 'after' stages of training for 'train', 'val', and
        'test' splits. It also logs the server's centralized evaluation
        results if available.
        """
        for split, client_side_test_flag, server_side_test_flag in [
            (
                "train",
                self.args.common.test.client.train,
                self.args.common.test.server.train,
            ),
            ("val", self.args.common.test.client.val, self.args.common.test.server.val),
            (
                "test",
                self.args.common.test.client.test,
                self.args.common.test.server.test,
            ),
        ]:
            for stage in ["before", "after"]:
                if client_side_test_flag:
                    aggregated = Metrics()
                    for i in self.selected_clients:
                        aggregated.update(
                            self.client_metrics[i][self.current_epoch][stage][split]
                        )

                    self.aggregated_client_metrics[stage][split].append(aggregated)

                    if self.args.common.monitor == "visdom":
                        self.viz.line(
                            [aggregated.accuracy],
                            [self.current_epoch],
                            win=f"Accuracy-{self.monitor_window_name_suffix}/{split}set-{stage}LocalTraining",
                            update="append",
                            name=self.algorithm_name,
                            opts=dict(
                                title=f"Accuracy-{self.monitor_window_name_suffix}/{split}set-{stage}LocalTraining",
                                xlabel="Communication Rounds",
                                ylabel="Accuracy",
                                legend=[self.algorithm_name],
                            ),
                        )
                    elif self.args.common.monitor == "tensorboard":
                        self.tensorboard.add_scalar(
                            f"Accuracy-{self.monitor_window_name_suffix}/{split}set-{stage}LocalTraining",
                            aggregated.accuracy,
                            self.current_epoch,
                            new_style=True,
                        )

            # log server side evaluation results
            if (
                server_side_test_flag
                and self.current_epoch + 1 in self.test_results
                and "centralized" in self.test_results[self.current_epoch + 1]
            ):
                if self.args.common.monitor == "visdom":
                    self.viz.line(
                        [
                            self.test_results[self.current_epoch + 1]["centralized"][
                                "after"
                            ][split].accuracy
                        ],
                        [self.current_epoch + 1],
                        win=f"Accuracy-{self.monitor_window_name_suffix}/{split}set-CentralizedEvaluation",
                        update="append",
                        name=self.algorithm_name,
                        opts=dict(
                            title=f"Accuracy-{self.monitor_window_name_suffix}/{split}set-CentralizedEvaluation",
                            xlabel="Communication Rounds",
                            ylabel="Accuracy",
                            legend=[self.algorithm_name],
                        ),
                    )
                elif self.args.common.monitor == "tensorboard":
                    self.tensorboard.add_scalar(
                        f"Accuracy-{self.monitor_window_name_suffix}/{split}set-CentralizedEvaluation",
                        self.test_results[self.current_epoch + 1]["centralized"][
                            "after"
                        ][split].accuracy,
                        self.current_epoch + 1,
                        new_style=True,
                    )

    def show_max_metrics(self):
        """Show the maximum stats that FL method get."""
        self.logger.log("=" * 20, self.algorithm_name, "Max Accuracy", "=" * 20)

        colors = {
            "before": "blue",
            "after": "red",
            "train": "yellow",
            "val": "green",
            "test": "cyan",
        }

        def _print(groups):
            for group in groups:
                epoches = [
                    E
                    for E, results in self.test_results.items()
                    if group in results.keys()
                ]
                if len(epoches) > 0:
                    self.logger.log(f"{group}:")
                    for stage in ["before", "after"]:
                        for split, flag in [
                            (
                                "train",
                                self.args.common.test.client.train
                                or self.args.common.test.server.train,
                            ),
                            (
                                "val",
                                self.args.common.test.client.val
                                or self.args.common.test.server.val,
                            ),
                            (
                                "test",
                                self.args.common.test.client.test
                                or self.args.common.test.server.test,
                            ),
                        ]:
                            if flag:
                                metrics_list = list(
                                    map(
                                        lambda E: (
                                            E,
                                            self.test_results[E][group][stage][split],
                                        ),
                                        epoches,
                                    )
                                )
                                if len(metrics_list) > 0:
                                    epoch, max_acc = max(
                                        [
                                            (epoch, metrics.accuracy)
                                            for epoch, metrics in metrics_list
                                        ],
                                        key=lambda tup: tup[1],
                                    )
                                    self.logger.log(
                                        f"[{colors[split]}]({split})[/{colors[split]}] "
                                        f"[{colors[stage]}]{stage}[/{colors[stage]}] "
                                        f"fine-tuning: {max_acc:.2f}% at epoch {epoch}"
                                    )

        if self.train_clients == self.val_clients == self.test_clients:
            _print(["all_clients"])
        else:
            _print(["val_clients", "test_clients"])
        if self.args.common.test.server.interval > 0:
            _print(["centralized"])

    def save_model_weights(self):
        model_name = f"{self.args.dataset.name}_{self.args.common.global_epoch}_{self.args.model}.pt"
        if not self.all_model_params_personalized:
            torch.save(self.public_model_params, self.output_dir / model_name)
        else:
            self.logger.warn(
                f"{self.algorithm_name}'s all_model_params_personalized = True # `True` indicates that clients have their own fullset of personalized model parameters., "
                "which does not support saving model parameters. "
                "So the saving is skipped."
            )

    def save_learning_curve_plot(self):
        """Save the learning curves of FL-bench experiment."""
        import matplotlib
        from matplotlib import pyplot as plt

        matplotlib.use("Agg")
        linestyle = {
            "before": {"train": "dotted", "val": "dashed", "test": "solid"},
            "after": {"train": "dotted", "val": "dashed", "test": "solid"},
        }
        for stage in ["before", "after"]:
            for split in ["train", "val", "test"]:
                if len(self.aggregated_client_metrics[stage][split]) > 0:
                    plt.plot(
                        [
                            metrics.accuracy
                            for metrics in self.aggregated_client_metrics[stage][split]
                        ],
                        label=f"{split}set ({stage}LocalTraining)",
                        ls=linestyle[stage][split],
                    )

        plt.title(f"{self.algorithm_name}_{self.args.dataset.name}")
        plt.ylim(0, 100)
        plt.xlabel("Communication Rounds")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(self.output_dir / f"metrics.png", bbox_inches="tight")

    def save_metrics_stats(self):
        """Save the metrics stats of FL-bench experiment."""
        import pandas as pd

        df = pd.DataFrame()
        for stage in ["before", "after"]:
            for split in ["train", "val", "test"]:
                if len(self.aggregated_client_metrics[stage][split]) > 0:
                    for metric in [
                        "accuracy",
                        # "micro_precision",
                        # "macro_precision",
                        # "micro_recall",
                        # "macro_recall",
                    ]:
                        stats = [
                            getattr(metrics, metric)
                            for metrics in self.aggregated_client_metrics[stage][split]
                        ]
                        df.insert(
                            loc=df.shape[1],
                            column=f"{metric}_{split}_{stage}",
                            value=np.array(stats).T,
                        )
        df.to_csv(self.output_dir / f"metrics.csv", index=True, index_label="epoch")

    def run_experiment(self):
        """The entrypoint of FL-bench experiment."""
        self.logger.log("=" * 20, self.algorithm_name, "=" * 20)
        self.logger.log("Experiment Arguments:")
        rich_pprint(
            OmegaConf.to_object(self.args), console=self.logger.stdout, expand_all=True
        )
        if self.args.common.save_log:
            rich_pprint(
                OmegaConf.to_object(self.args),
                console=self.logger.logfile_logger,
                expand_all=True,
            )
        if self.args.common.monitor == "tensorboard":
            self.tensorboard.add_text(
                f"ExperimentalArguments-{self.monitor_window_name_suffix}",
                f"{json.dumps(OmegaConf.to_object(self.args), indent=4)}",
            )

        begin = time.time()
        try:
            self.train()
        except KeyboardInterrupt:
            # when user manually terminates the run, FL-bench
            # indicates that run should be considered as useless and deleted.
            self.logger.close()
            del self.train_progress_bar
            if self.args.common.delete_useless_run:
                if os.path.isdir(self.output_dir):
                    shutil.rmtree(self.output_dir)
                return
        except Exception as e:
            self.logger.log(traceback.format_exc())
            self.logger.log(f"Exception occurred: {e}")
            self.logger.close()
            del self.train_progress_bar
            raise

        end = time.time()
        total = end - begin
        self.logger.log(
            f"{self.algorithm_name}'s total running time: "
            f"{int(total // 3600)} h {int((total % 3600) // 60)} m {int(total % 60)} s."
        )
        self.logger.log("=" * 20, self.algorithm_name, "Experiment Results:", "=" * 20)
        self.logger.log(
            "[green]Display format: (before local fine-tuning) -> (after local fine-tuning)\n",
            "So if finetune_epoch = 0, x.xx% -> 0.00% is normal.\n",
            "Centralized testing ONLY happens after model aggregation, so the stats between '->' are the same.",
        )
        all_test_results = {
            epoch: {
                group: {
                    split: {
                        "loss": f"[red]{metrics['before'][split].loss:.4f} -> "
                        f"{metrics['after'][split].loss:.4f}[/red]",
                        "accuracy": f"[blue]{metrics['before'][split].accuracy:.2f}% -> "
                        f"{metrics['after'][split].accuracy:.2f}%[/blue]",
                    }
                    for split, flag in [
                        (
                            "train",
                            self.args.common.test.client.train
                            or self.args.common.test.server.train,
                        ),
                        (
                            "val",
                            self.args.common.test.client.val
                            or self.args.common.test.server.val,
                        ),
                        (
                            "test",
                            self.args.common.test.client.test
                            or self.args.common.test.server.test,
                        ),
                    ]
                    if flag
                }
                for group, metrics in results.items()
            }
            for epoch, results in self.test_results.items()
        }

        self.logger.log(json.dumps(all_test_results, indent=4))
        if self.args.common.monitor == "tensorboard":
            for epoch, results in all_test_results.items():
                self.tensorboard.add_text(
                    f"Results-{self.monitor_window_name_suffix}",
                    text_string=f"<pre>{results}</pre>",
                    global_step=epoch,
                )

        self.show_max_metrics()

        self.logger.close()

        # plot the training curves
        if self.args.common.save_learning_curve_plot:
            self.save_learning_curve_plot()

        # save each round's metrics stats
        if self.args.common.save_metrics:
            self.save_metrics_stats()

        # save trained model(s) parameters
        if self.args.common.save_model:
            self.save_model_weights()
