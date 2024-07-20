import functools
import inspect
import json
import os
import pickle
import random
import time
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import numpy as np
import ray
import torch
from rich.console import Console
from rich.json import JSON
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
    OUT_DIR,
)
from src.utils.metrics import Metrics
from src.utils.models import MODELS, DecoupledModel
from src.utils.tools import (
    Logger,
    NestedNamespace,
    fix_random_seed,
    get_optimal_cuda_device,
)
from src.utils.trainer import FLbenchTrainer


class FedAvgServer:
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedAvg",
        unique_model=False,
        use_fedavg_client_cls=True,
        return_diff=False,
    ):
        """
        Args:
            `args`: A nested Namespace object of the arguments.
            `algo`: Name of FL method.
            `unique_model`: `True` indicates that clients have their own fullset model parameters.
            `use_fedavg_client_cls`: `True` indicates that using default `FedAvgClient()` as the client class.
            `return_diff`: `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
        """
        self.args = args
        self.algo = algo
        self.unique_model = unique_model
        self.return_diff = return_diff
        fix_random_seed(self.args.common.seed)
        start_time = str(
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(round(time.time())))
        )
        self.output_dir = OUT_DIR / self.algo / start_time
        with open(
            FLBENCH_ROOT / "data" / self.args.common.dataset / "args.json", "r"
        ) as f:
            self.args.dataset = NestedNamespace(json.load(f))

        # get client party info
        try:
            partition_path = (
                FLBENCH_ROOT / "data" / self.args.common.dataset / "partition.pkl"
            )
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")
        self.train_clients: list[int] = partition["separation"]["train"]
        self.test_clients: list[int] = partition["separation"]["test"]
        self.val_clients: list[int] = partition["separation"]["val"]
        self.client_num: int = partition["separation"]["total"]

        # init model(s) parameters
        self.device = get_optimal_cuda_device(self.args.common.use_cuda)

        # get_model_arch() would return a class depends on model's name,
        # then init the model object by indicating the dataset and calling the class.
        # Finally transfer the model object to the target device.
        self.model: DecoupledModel = MODELS[self.args.common.model](
            dataset=self.args.common.dataset
        )
        self.model.check_and_preprocess(self.args)

        _init_global_params, _init_global_params_name = [], []
        for key, param in self.model.named_parameters():
            _init_global_params.append(param.data.clone())
            _init_global_params_name.append(key)

        self.public_model_param_names = _init_global_params_name
        self.public_model_params: OrderedDict[str, torch.Tensor] = OrderedDict(
            zip(_init_global_params_name, _init_global_params)
        )

        if self.args.common.external_model_params_file is not None:
            file_path = str(
                (FLBENCH_ROOT / self.args.common.external_model_params_file).absolute()
            )
            if os.path.isfile(file_path) and file_path.find(".pt") != -1:
                external_params = torch.load(file_path, map_location="cpu")
                self.public_model_params.update(external_params)
            elif not os.path.isfile(file_path):
                raise FileNotFoundError(f"{file_path} is not a valid file path.")
            elif file_path.find(".pt") == -1:
                raise TypeError(f"{file_path} is not a valid .pt file.")

        self.clients_personal_model_params = {i: {} for i in range(self.client_num)}

        if self.args.common.buffers == "local":
            _init_buffers = OrderedDict(self.model.named_buffers())
            for i in range(self.client_num):
                self.clients_personal_model_params[i] = deepcopy(_init_buffers)

        if self.unique_model:
            for params_dict in self.clients_personal_model_params.values():
                params_dict.update(deepcopy(self.model.state_dict()))

        self.client_optimizer_states = {i: {} for i in range(self.client_num)}

        self.client_lr_scheduler_states = {i: {} for i in range(self.client_num)}

        self.client_local_epoches: list[int] = [
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
        self.selected_clients: list[int] = []
        self.current_epoch = 0

        # For controlling behaviors of some specific methods while testing (not used by all methods)
        self.testing = False

        if not os.path.isdir(self.output_dir) and (
            self.args.common.save_log
            or self.args.common.save_fig
            or self.args.common.save_metrics
        ):
            os.makedirs(self.output_dir, exist_ok=True)

        self.client_metrics = {i: {} for i in self.train_clients}
        self.global_metrics = {
            "before": {"train": [], "val": [], "test": []},
            "after": {"train": [], "val": [], "test": []},
        }

        self.verbose = False
        stdout = Console(log_path=False, log_time=False, soft_wrap=True, tab_size=4)
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.common.save_log,
            logfile_path=OUT_DIR
            / self.algo
            / self.output_dir
            / f"{self.args.common.dataset}.log",
        )
        self.test_results: dict[int, dict[str, dict[str, Metrics]]] = {}
        self.train_progress_bar = track(
            range(self.args.common.global_epoch),
            "[bold green]Training...",
            console=stdout,
        )

        if self.args.common.visible is not None:
            self.monitor_window_name_suffix = (
                self.args.dataset.monitor_window_name_suffix
            )

        if self.args.common.visible == "visdom":
            from visdom import Visdom

            self.viz = Visdom()
        elif self.args.common.visible == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(log_dir=self.output_dir)

        # init trainer
        self.trainer: FLbenchTrainer = None
        if use_fedavg_client_cls:
            self.init_trainer()

    def init_trainer(self, fl_client_cls=FedAvgClient, **extras):
        """Initiate the FL-bench trainier that responsible to client training.
        `extras` are the arguments of `fl_client_cls.__init__()` that not in
        `[model, args, optimizer_cls, lr_scheduler_cls, dataset, data_indices,
        device, return_diff]`, which are essential for all methods in FL-bench.

        Args:
            `fl_client_cls`: The class of client in FL method. Defaults to `FedAvgClient`.
        """
        if self.args.mode == "serial" or self.args.parallel.num_workers < 2:
            self.trainer = FLbenchTrainer(
                server=self,
                client_cls=fl_client_cls,
                mode=MODE.SERIAL,
                num_workers=0,
                init_args=dict(
                    model=deepcopy(self.model),
                    optimizer_cls=self.get_client_optimizer(),
                    lr_scheduler_cls=self.get_client_lr_scheduler(),
                    args=self.args,
                    dataset=self.get_dataset(),
                    data_indices=self.get_clients_data_indices(),
                    device=self.device,
                    return_diff=self.return_diff,
                    **extras,
                ),
            )
        else:
            model_ref = ray.put(self.model.cpu())
            optimzier_cls_ref = ray.put(self.get_client_optimizer())
            lr_scheduler_cls_ref = ray.put(self.get_client_lr_scheduler())
            dataset_ref = ray.put(self.get_dataset())
            data_indices_ref = ray.put(self.get_clients_data_indices())
            args_ref = ray.put(self.args)
            device_ref = ray.put(None)  # in parallel mode, workers decide their device
            return_diff_ref = ray.put(self.return_diff)
            self.trainer = FLbenchTrainer(
                server=self,
                client_cls=fl_client_cls,
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
                FLBENCH_ROOT / "data" / self.args.common.dataset / "partition.pkl"
            )
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(
                f"Please partition {self.args.common.dataset} first."
            )

        # [0: {"train": [...], "val": [...], "test": [...]}, ...]
        data_indices: list[dict[str, list[int]]] = partition["data_indices"]

        return data_indices

    def get_dataset(self) -> BaseDataset:
        """Load the specified dataset according to the configuration.

        Returns:
        BaseDataset: This is the loaded dataset instance,
        which inherits from the BaseDataset class.
        """
        dataset: BaseDataset = DATASETS[self.args.common.dataset](
            root=FLBENCH_ROOT / "data" / self.args.common.dataset,
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
                    DATA_MEAN[self.args.common.dataset],
                    DATA_STD[self.args.common.dataset],
                )
            ]
            if self.args.common.dataset in DATA_MEAN
            and self.args.common.dataset in DATA_STD
            else []
        )
        test_target_transform = transforms.Compose([])
        train_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[self.args.common.dataset],
                    DATA_STD[self.args.common.dataset],
                )
            ]
            if self.args.common.dataset in DATA_MEAN
            and self.args.common.dataset in DATA_STD
            else []
        )
        train_target_transform = transforms.Compose([])
        return dict(
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
        )

    def get_client_optimizer(self):
        """Get client-side model training optimizer.

        Returns:
            A partial initiated optimizer class that client only need to add `params` arg.
        """
        target_optimizer_cls: type[torch.optim.Optimizer] = OPTIMIZERS[
            self.args.common.optimizer.name
        ]
        keys_required = inspect.getfullargspec(target_optimizer_cls.__init__).args
        args_valid = {}
        for key, value in vars(self.args.common.optimizer).items():
            if key in keys_required:
                args_valid[key] = value

        optimizer_cls = functools.partial(target_optimizer_cls, **args_valid)
        args_valid["name"] = self.args.common.optimizer.name
        self.args.common.optimizer = NestedNamespace(args_valid)
        return optimizer_cls

    def get_client_lr_scheduler(self):
        if hasattr(self.args.common, "lr_scheduler"):
            if self.args.common.lr_scheduler.name is None:
                del self.args.common.lr_scheduler
                return None
            lr_scheduler_args = getattr(self.args.common, "lr_scheduler")
            if lr_scheduler_args.name is not None:
                target_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] = (
                    LR_SCHEDULERS[lr_scheduler_args.name]
                )
                keys_required = inspect.getfullargspec(
                    target_scheduler_cls.__init__
                ).args

                args_valid = {}
                for key, value in vars(self.args.common.lr_scheduler).items():
                    if key in keys_required:
                        args_valid[key] = value

                lr_scheduler_cls = functools.partial(target_scheduler_cls, **args_valid)
                args_valid["name"] = self.args.common.lr_scheduler.name
                self.args.common.lr_scheduler = NestedNamespace(args_valid)
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
            self.log_info()
            avg_round_time = (avg_round_time * (self.current_epoch) + (end - begin)) / (
                self.current_epoch + 1
            )

            if (E + 1) % self.args.common.test_interval == 0:
                self.test()

        self.logger.log(
            f"{self.algo}'s average time taken by each global epoch: "
            f"{int(avg_round_time // 60)} min {(avg_round_time % 60):.2f} sec."
        )

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at
        server side) in each communication round."""

        client_packages = self.trainer.train()
        self.aggregate(client_packages)

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

    def test(self):
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

            self.test_results[self.current_epoch + 1] = results

        self.testing = False

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
    def aggregate(self, client_packages: OrderedDict[int, dict[str, Any]]):
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

    def show_convergence(self):
        """Collect the number of epoches that FL method reach specific
        accuracies while training."""
        colors = {
            "before": "blue",
            "after": "red",
            "train": "yellow",
            "val": "green",
            "test": "cyan",
        }
        self.logger.log("=" * 10, self.algo, "Convergence on train clients", "=" * 10)
        for stage in ["before", "after"]:
            for split in ["train", "val", "test"]:
                if len(self.global_metrics[stage][split]) > 0:
                    self.logger.log(
                        f"[{colors[split]}]{split}[/{colors[split]}] "
                        f"[{colors[stage]}]({stage} local training):"
                    )
                    acc_range = [90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
                    min_acc_idx = 10
                    max_acc = 0
                    accuracies = [
                        metrics.accuracy
                        for metrics in self.global_metrics[stage][split]
                    ]
                    for E, acc in enumerate(accuracies):
                        for i, target in enumerate(acc_range):
                            if acc >= target and acc > max_acc:
                                self.logger.log(f"{target}%({acc:.2f}%) at epoch: {E}")
                                max_acc = acc
                                min_acc_idx = i
                                break
                        acc_range = acc_range[:min_acc_idx]

    def log_info(self):
        """Accumulate client evaluation results at each round."""
        for stage in ["before", "after"]:
            for split, flag in [
                ("train", self.args.common.eval_train),
                ("val", self.args.common.eval_val),
                ("test", self.args.common.eval_test),
            ]:
                if flag:
                    global_metrics = Metrics()
                    for i in self.selected_clients:
                        global_metrics.update(
                            self.client_metrics[i][self.current_epoch][stage][split]
                        )

                    self.global_metrics[stage][split].append(global_metrics)

                    if self.args.common.visible == "visdom":
                        self.viz.line(
                            [global_metrics.accuracy],
                            [self.current_epoch],
                            win=f"Accuracy-{self.monitor_window_name_suffix}/{split}set-{stage}LocalTraining",
                            update="append",
                            name=self.algo,
                            opts=dict(
                                title=f"Accuracy-{self.monitor_window_name_suffix}/{split}set-{stage}LocalTraining",
                                xlabel="Communication Rounds",
                                ylabel="Accuracy",
                                legend=[self.algo],
                            ),
                        )
                    elif self.args.common.visible == "tensorboard":
                        self.tensorboard.add_scalar(
                            f"Accuracy-{self.monitor_window_name_suffix}/{split}set-{stage}LocalTraining",
                            global_metrics.accuracy,
                            self.current_epoch,
                            new_style=True,
                        )

    def show_max_metrics(self):
        """Show the maximum stats that FL method get."""
        self.logger.log("=" * 20, self.algo, "Max Accuracy", "=" * 20)

        colors = {
            "before": "blue",
            "after": "red",
            "train": "yellow",
            "val": "green",
            "test": "cyan",
        }

        groups = ["val_clients", "test_clients"]
        if self.train_clients == self.val_clients == self.test_clients:
            groups = ["all_clients"]

        for group in groups:
            self.logger.log(f"{group}:")
            for stage in ["before", "after"]:
                for split, flag in [
                    ("train", self.args.common.eval_train),
                    ("val", self.args.common.eval_val),
                    ("test", self.args.common.eval_test),
                ]:
                    if flag:
                        metrics_list = list(
                            map(
                                lambda tup: (tup[0], tup[1][group][stage][split]),
                                self.test_results.items(),
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

    def run(self):
        """The entrypoint of FL-bench experiment.

        Raises:
            RuntimeError: When FL-bench trainer is not set properly.
        """
        self.logger.log("=" * 20, self.algo, "=" * 20)
        self.logger.log("Experiment Arguments:")
        self.logger.log(JSON(str(self.args)))
        if self.args.common.visible == "tensorboard":
            self.tensorboard.add_text(
                f"ExperimentalArguments-{self.monitor_window_name_suffix}",
                f"<pre>{self.args}</pre>",
            )

        begin = time.time()
        try:
            self.train()
        except KeyboardInterrupt:
            # when user press Ctrl+C
            # indicates that this run should be considered as useless and deleted.
            self.logger.close()
            del self.train_progress_bar
            if self.args.common.delete_useless_run:
                if os.path.isdir(self.output_dir):
                    os.removedirs(self.output_dir)
                return
        except:
            self.logger.close()
            del self.train_progress_bar
            raise

        end = time.time()
        total = end - begin
        self.logger.log(
            f"{self.algo}'s total running time: "
            f"{int(total // 3600)} h {int((total % 3600) // 60)} m {int(total % 60)} s."
        )
        self.logger.log("=" * 20, self.algo, "Experiment Results:", "=" * 20)
        self.logger.log(
            "Format: [green](before local fine-tuning) -> [blue](after local fine-tuning)\n",
            "So if finetune_epoch = 0, x.xx% -> 0.00% is normal.",
        )
        all_test_results = {
            epoch: {
                group: {
                    split: {
                        "loss": f"{metrics['before'][split].loss:.4f} -> {metrics['after'][split].loss:.4f}",
                        "accuracy": f"{metrics['before'][split].accuracy:.2f}% -> {metrics['after'][split].accuracy:.2f}%",
                    }
                    for split, flag in [
                        ("train", self.args.common.eval_train),
                        ("val", self.args.common.eval_val),
                        ("test", self.args.common.eval_test),
                    ]
                    if flag
                }
                for group, metrics in results.items()
            }
            for epoch, results in self.test_results.items()
        }

        self.logger.log(all_test_results)
        if self.args.common.visible == "tensorboard":
            for epoch, results in all_test_results.items():
                self.tensorboard.add_text(
                    f"Results-{self.monitor_window_name_suffix}",
                    text_string=f"<pre>{results}</pre>",
                    global_step=epoch,
                )

        self.show_convergence()
        self.show_max_metrics()

        self.logger.close()

        # plot the training curves
        if self.args.common.save_fig:
            import matplotlib
            from matplotlib import pyplot as plt

            matplotlib.use("Agg")
            linestyle = {
                "before": {"train": "dotted", "val": "dashed", "test": "solid"},
                "after": {"train": "dotted", "val": "dashed", "test": "solid"},
            }
            for stage in ["before", "after"]:
                for split in ["train", "val", "test"]:
                    if len(self.global_metrics[stage][split]) > 0:
                        plt.plot(
                            [
                                metrics.accuracy
                                for metrics in self.global_metrics[stage][split]
                            ],
                            label=f"{split}set ({stage}LocalTraining)",
                            ls=linestyle[stage][split],
                        )

            plt.title(f"{self.algo}_{self.args.common.dataset}")
            plt.ylim(0, 100)
            plt.xlabel("Communication Rounds")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(
                OUT_DIR
                / self.algo
                / self.output_dir
                / f"{self.args.common.dataset}.png",
                bbox_inches="tight",
            )

        # save each round's metrics stats
        if self.args.common.save_metrics:
            import pandas as pd

            df = pd.DataFrame()
            for stage in ["before", "after"]:
                for split in ["train", "val", "test"]:
                    if len(self.global_metrics[stage][split]) > 0:
                        for metric in [
                            "accuracy",
                            # "micro_precision",
                            # "macro_precision",
                            # "micro_recall",
                            # "macro_recall",
                        ]:
                            stats = [
                                getattr(metrics, metric)
                                for metrics in self.global_metrics[stage][split]
                            ]
                            df.insert(
                                loc=df.shape[1],
                                column=f"{metric}_{split}_{stage}",
                                value=np.array(stats).T,
                            )
            df.to_csv(
                OUT_DIR
                / self.algo
                / self.output_dir
                / f"{self.args.common.dataset}_acc_metrics.csv",
                index=True,
                index_label="epoch",
            )

        # save trained model(s) parameters
        if self.args.common.save_model:
            model_name = f"{self.args.common.dataset}_{self.args.common.global_epoch}_{self.args.common.model}.pt"
            if not self.unique_model:
                torch.save(self.public_model_params, self.output_dir / model_name)
            else:
                warnings.warn(
                    f"{self.algo}'s unique_model = True, which does not support saving model parameters. "
                    "So the saving is skipped."
                )
