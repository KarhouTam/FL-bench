import pickle
import json
import os
import time
import random
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict

import torch
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.json import JSON

from src.utils.tools import (
    Logger,
    NestedNamespace,
    fix_random_seed,
    trainable_params,
    get_optimal_cuda_device,
)
from src.utils.models import MODELS, DecoupledModel
from src.utils.metrics import Metrics
from src.client.fedavg import FedAvgClient
from src.utils.constants import FLBENCH_ROOT, OUT_DIR

class FedAvgServer:
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedAvg",
        unique_model=False,
        default_trainer=True,
    ):
        self.args = args
        self.algo = algo
        self.unique_model = unique_model
        fix_random_seed(self.args.common.seed)
        start_time = str(
            time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(round(time.time())))
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
        self.train_clients: List[int] = partition["separation"]["train"]
        self.test_clients: List[int] = partition["separation"]["test"]
        self.val_clients: List[int] = partition["separation"]["val"]
        self.client_num: int = partition["separation"]["total"]

        # init model(s) parameters
        self.device = get_optimal_cuda_device(self.args.common.use_cuda)

        # get_model_arch() would return a class depends on model's name,
        # then init the model object by indicating the dataset and calling the class.
        # Finally transfer the model object to the target device.
        self.model: DecoupledModel = MODELS[self.args.common.model](
            dataset=self.args.common.dataset
        ).to(self.device)
        self.model.check_avaliability()

        # client_trainable_params is for pFL, which outputs exclusive model per client
        # global_params_dict is for traditional FL, which outputs a single global model
        self.client_trainable_params: List[List[torch.Tensor]] = None
        self.global_params_dict: OrderedDict[str, torch.Tensor] = None

        random_init_params, self.trainable_params_name = trainable_params(
            self.model, detach=True, requires_name=True
        )
        self.global_params_dict = OrderedDict(
            zip(self.trainable_params_name, random_init_params)
        )
        if (
            not self.unique_model
            and self.args.common.external_model_params_file
            and os.path.isfile(self.args.common.external_model_params_file)
        ):
            # load pretrained params
            self.global_params_dict = torch.load(
                self.args.common.external_model_params_file, map_location=self.device
            )
        else:
            self.client_trainable_params = [
                trainable_params(self.model, detach=True) for _ in self.train_clients
            ]

        # system heterogeneity (straggler) setting
        self.clients_local_epoch: List[int] = [
            self.args.common.local_epoch
        ] * self.client_num
        if (
            self.args.common.straggler_ratio > 0
            and self.args.common.local_epoch
            > self.args.common.straggler_min_local_epoch
        ):
            straggler_num = int(self.client_num * self.args.common.straggler_ratio)
            normal_num = self.client_num - straggler_num
            self.clients_local_epoch = [self.args.common.local_epoch] * (
                normal_num
            ) + random.choices(
                range(
                    self.args.common.straggler_min_local_epoch,
                    self.args.common.local_epoch,
                ),
                k=straggler_num,
            )
            random.shuffle(self.clients_local_epoch)

        # To make sure all algorithms run through the same client sampling stream.
        # Some algorithms' implicit operations at client side may disturb the stream if sampling happens at each FL round's beginning.
        self.client_sample_stream = [
            random.sample(
                self.train_clients,
                max(1, int(self.client_num * self.args.common.join_ratio)),
            )
            for _ in range(self.args.common.global_epoch)
        ]
        self.selected_clients: List[int] = []
        self.current_epoch = 0
        # epoch that need test model on test clients.
        self.epoch_test = [
            epoch
            for epoch in range(0, self.args.common.global_epoch)
            if (epoch + 1) % self.args.common.test_interval == 0
        ]
        # For controlling behaviors of some specific methods while testing (not used by all methods)
        self.testing = False

        if not os.path.isdir(self.output_dir) and (
            self.args.common.save_log
            or self.args.common.save_fig
            or self.args.common.save_metrics
        ):
            os.makedirs(self.output_dir, exist_ok=True)

        if self.args.common.visible:
            from visdom import Visdom

            self.viz = Visdom()
            if self.args.common.viz_win_name is not None:
                self.viz_win_name = self.args.common.viz_win_name
            else:
                self.viz_win_name = (
                    f"{self.algo}"
                    + f"_{self.args.common.dataset}"
                    + f"_{self.args.common.global_epoch}"
                    + f"_{self.args.common.local_epoch}"
                )
        self.client_metrics = {i: {} for i in self.train_clients}
        self.global_metrics = {
            "before": {"train": [], "val": [], "test": []},
            "after": {"train": [], "val": [], "test": []},
        }
        stdout = Console(log_path=False, log_time=False)
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.common.save_log,
            logfile_path=OUT_DIR
            / self.algo
            / self.output_dir
            / f"{self.args.common.dataset}_log.html",
        )
        self.test_results: Dict[int, Dict[str, Dict[str, Metrics]]] = {}
        self.train_progress_bar = track(
            range(self.args.common.global_epoch),
            "[bold green]Training...",
            console=stdout,
        )

        # init trainer
        self.trainer = None
        if default_trainer:
            self.trainer = FedAvgClient(
                deepcopy(self.model), self.args, self.logger, self.device
            )

        self.logger.log("=" * 20, self.algo, "=" * 20)
        self.logger.log("Experiment Arguments:")
        self.logger.log(JSON(str(self.args)))

    def train(self):
        """The Generic FL training process"""
        avg_round_time = 0
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.common.verbose_gap == 0:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E + 1}", "-" * 26)

            if (E + 1) % self.args.common.test_interval == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]
            begin = time.time()
            self.train_one_round()
            end = time.time()
            self.log_info()
            avg_round_time = (avg_round_time * (self.current_epoch) + (end - begin)) / (
                self.current_epoch + 1
            )

        self.logger.log(
            f"{self.algo}'s average time taken by each global epoch: {int(avg_round_time // 60)} m {(avg_round_time % 60):.2f} s."
        )

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        delta_cache = []
        weight_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            (delta, weight, self.client_metrics[client_id][self.current_epoch]) = (
                self.trainer.train(
                    client_id=client_id,
                    local_epoch=self.clients_local_epoch[client_id],
                    new_parameters=client_local_params,
                    verbose=((self.current_epoch + 1) % self.args.common.verbose_gap)
                    == 0,
                )
            )

            delta_cache.append(delta)
            weight_cache.append(weight)

        self.aggregate(delta_cache, weight_cache)

    def test(self):
        """The function for testing FL method's output (a single global model or personalized client models)."""
        self.testing = True
        client_ids = set(self.val_clients + self.test_clients)
        all_same = False
        if client_ids:
            if self.val_clients == self.train_clients == self.test_clients:
                all_same = True
                results = {
                    "all_clients": {
                        "before": {
                            "train": Metrics(),
                            "val": Metrics(),
                            "test": Metrics(),
                        },
                        "after": {
                            "train": Metrics(),
                            "val": Metrics(),
                            "test": Metrics(),
                        },
                    }
                }
            else:
                results = {
                    "val_clients": {
                        "before": {
                            "train": Metrics(),
                            "val": Metrics(),
                            "test": Metrics(),
                        },
                        "after": {
                            "train": Metrics(),
                            "val": Metrics(),
                            "test": Metrics(),
                        },
                    },
                    "test_clients": {
                        "before": {
                            "train": Metrics(),
                            "val": Metrics(),
                            "test": Metrics(),
                        },
                        "after": {
                            "train": Metrics(),
                            "val": Metrics(),
                            "test": Metrics(),
                        },
                    },
                }
            for cid in client_ids:
                client_local_params = self.generate_client_params(cid)
                client_metrics = self.trainer.test(cid, client_local_params)

                for stage in ["before", "after"]:
                    for split in ["train", "val", "test"]:
                        if all_same:
                            results["all_clients"][stage][split].update(
                                client_metrics[stage][split]
                            )
                        else:
                            if cid in self.val_clients:
                                results["val_clients"][stage][split].update(
                                    client_metrics[stage][split]
                                )
                            if cid in self.test_clients:
                                results["test_clients"][stage][split].update(
                                    client_metrics[stage][split]
                                )

            self.test_results[self.current_epoch + 1] = results

        self.testing = False

    @torch.no_grad()
    def update_client_params(self, client_params_cache: List[List[torch.Tensor]]):
        """
        The function for updating clients model while unique_model is `True`.
        This function is only useful for some pFL methods.

        Args:
            client_params_cache (List[List[torch.Tensor]]): models parameters of selected clients.

        Raises:
            RuntimeError: If unique_model = `False`, this function will not work properly.
        """
        if self.unique_model:
            for i, client_id in enumerate(self.selected_clients):
                self.client_trainable_params[client_id] = client_params_cache[i]
        else:
            raise RuntimeError(
                "FL system don't preserve params for each client (unique_model = False)."
            )

    def generate_client_params(self, client_id: int) -> OrderedDict[str, torch.Tensor]:
        """
        This function is for outputting model parameters that asked by `client_id`.

        Args:
            client_id (int): The ID of query client.

        Returns:
            OrderedDict[str, torch.Tensor]: The trainable model parameters.
        """
        if self.unique_model:
            return OrderedDict(
                zip(self.trainable_params_name, self.client_trainable_params[client_id])
            )
        else:
            return self.global_params_dict

    @torch.no_grad()
    def aggregate(
        self,
        delta_cache: List[OrderedDict[str, torch.Tensor]],
        weight_cache: List[int],
        return_diff=True,
    ):
        """
        This function is for aggregating recevied model parameters from selected clients.
        The method of aggregation is weighted averaging by default.

        Args:
            delta_cache (List[List[torch.Tensor]]): `delta` means the difference between client model parameters that before and after local training.

            weight_cache (List[int]): Weight for each `delta` (client dataset size by default).

            return_diff (bool): Differnt value brings different operations. Default to True.
        """
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        if return_diff:
            delta_list = [list(delta.values()) for delta in delta_cache]
            aggregated_delta = [
                torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
                for diff in zip(*delta_list)
            ]

            for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
                param.data -= diff
        else:
            for old_param, zipped_new_param in zip(
                self.global_params_dict.values(), zip(*delta_cache)
            ):
                old_param.data = (torch.stack(zipped_new_param, dim=-1) * weights).sum(
                    dim=-1
                )
        self.model.load_state_dict(self.global_params_dict, strict=False)

    def show_convergence(self):
        """This function is for checking model convergence through the entire FL training process."""
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
                        f"[{colors[split]}]{split}[/{colors[split]}] [{colors[stage]}]({stage} local training):"
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
        """This function is for logging each selected client's training info."""
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

                    if self.args.common.visible:
                        self.viz.line(
                            [global_metrics.accuracy],
                            [self.current_epoch],
                            win=self.viz_win_name,
                            update="append",
                            name=f"{split}({stage})",
                            opts=dict(
                                title=self.viz_win_name,
                                xlabel="Communication Rounds",
                                ylabel="Accuracy",
                            ),
                        )

    def log_max_metrics(self):
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
                                f"[{colors[split]}]({split})[/{colors[split]}] [{colors[stage]}]{stage}[/{colors[stage]}] fine-tuning: {max_acc:.2f}% at epoch {epoch}"
                            )

    def run(self):
        """The comprehensive FL process.

        Raises:
            RuntimeError: If `trainer` is not set.
        """
        begin = time.time()
        if self.trainer is None:
            raise RuntimeError(
                "Specify your unique trainer or set `default_trainer` as True."
            )

        if self.args.common.visible:
            self.viz.close(win=self.viz_win_name)

        self.train()
        end = time.time()
        total = end - begin
        self.logger.log(
            f"{self.algo}'s total running time: {int(total // 3600)} h {int((total % 3600) // 60)} m {int(total % 60)} s."
        )
        self.logger.log("=" * 20, self.algo, "Experiment Results:", "=" * 20)
        self.logger.log(
            "Format: [green](before local fine-tuning) -> [blue](after local fine-tuning)"
        )
        self.logger.log(
            {
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
        )

        if self.args.common.check_convergence:
            self.show_convergence()
        self.log_max_metrics()
        self.logger.close()

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
                            label=f"{split}_{stage}",
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
                / f"{self.args.common.dataset}.pdf",
                bbox_inches="tight",
            )
        if self.args.common.save_metrics:
            import pandas as pd

            df = pd.DataFrame()
            for stage in ["before", "after"]:
                for split in ["train", "val", "test"]:
                    if len(self.global_metrics[stage][split]) > 0:
                        for metric in [
                            "accuracy",
                            "micro_precision",
                            "macro_precision",
                            "micro_recall",
                            "macro_recall",
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
        # save trained model(s)
        if self.args.common.save_model:
            model_name = f"{self.args.common.dataset}_{self.args.common.global_epoch}_{self.args.common.model}.pt"
            if self.unique_model:
                torch.save(self.client_trainable_params, self.output_dir / model_name)
            else:
                torch.save(self.global_params_dict, self.output_dir / model_name)
