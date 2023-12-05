import pickle
import sys
import json
import os
import random
from pathlib import Path
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
import time
from typing import Dict, List, OrderedDict

import torch
import numpy as np
from rich.console import Console
from rich.progress import track

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

sys.path.append(PROJECT_DIR.as_posix())

from src.utils.tools import (
    OUT_DIR,
    Logger,
    fix_random_seed,
    parse_config_file,
    trainable_params,
    get_best_device,
)
from src.utils.models import get_model_arch
from src.client.fedavg import FedAvgClient


def local_time():
    now = int(round(time.time() * 1000))
    now02 = time.strftime(
        "%Y-%m-%d-%H:%M:%S", time.localtime(now / 1000)
    )  # e.g. 2023-11-08-10:31:47
    return now02


def get_fedavg_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="lenet5",
        choices=[
            "lenet5",
            "avgcnn",
            "alex",
            "2nn",
            "squeeze0",
            "squeeze1",
            "res18",
            "res34",
            "res50",
            "res101",
            "res152",
            "dense121",
            "dense161",
            "dense169",
            "dense201",
            "mobile2",
            "mobile3s",
            "mobile3l",
            "efficient0",
            "efficient1",
            "efficient2",
            "efficient3",
            "efficient4",
            "efficient5",
            "efficient6",
            "efficient7",
            "custom",
        ],
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "cifar10",
            "cifar100",
            "synthetic",
            "femnist",
            "emnist",
            "fmnist",
            "celeba",
            "medmnistS",
            "medmnistA",
            "medmnistC",
            "covid19",
            "svhn",
            "usps",
            "tiny_imagenet",
            "cinic10",
            "domain",
        ],
        default="cifar10",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-jr", "--join_ratio", type=float, default=0.1)
    parser.add_argument("-ge", "--global_epoch", type=int, default=100)
    parser.add_argument("-le", "--local_epoch", type=int, default=5)
    parser.add_argument("-fe", "--finetune_epoch", type=int, default=0)
    parser.add_argument("-tg", "--test_gap", type=int, default=100)
    parser.add_argument("-ee", "--eval_test", type=int, default=1)
    parser.add_argument("-er", "--eval_train", type=int, default=0)
    parser.add_argument(
        "-op", "--optimizer", type=str, default="sgd", choices=["sgd", "adam"]
    )
    parser.add_argument("-lr", "--local_lr", type=float, default=1e-2)
    parser.add_argument("-mom", "--momentum", type=float, default=0.0)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-vg", "--verbose_gap", type=int, default=10)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-v", "--visible", type=int, default=0)
    parser.add_argument("--global_testset", type=int, default=0)
    parser.add_argument("--straggler_ratio", type=float, default=0)
    parser.add_argument("--straggler_min_local_epoch", type=int, default=1)
    parser.add_argument("--external_model_params_file", type=str, default="")
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--save_log", type=int, default=1)
    parser.add_argument("--save_model", type=int, default=0)
    parser.add_argument("--save_fig", type=int, default=1)
    parser.add_argument("--save_metrics", type=int, default=1)
    parser.add_argument("--viz_win_name", type=str, required=False)
    parser.add_argument("-cfg", "--config_file", type=str, default="")
    parser.add_argument("--check_convergence", type=int, default=1)
    return parser


class FedAvgServer:
    def __init__(
        self,
        algo: str = "FedAvg",
        args: Namespace = None,
        unique_model=False,
        default_trainer=True,
    ):
        self.args = get_fedavg_argparser().parse_args() if args is None else args
        self.algo = algo
        self.unique_model = unique_model
        if len(self.args.config_file) > 0 and os.path.exists(
            Path(self.args.config_file).absolute()
        ):
            self.args = parse_config_file(self.args)
        fix_random_seed(self.args.seed)

        # generate the name for output dir, e.g. "tau_0.1_mu_2.5_2023-12-04-20:30:49" for MOON.
        begin_time = str(local_time())
        known_args, unknown_args = get_fedavg_argparser().parse_known_args()
        args_original_dict = vars(known_args)
        args_all_dict = vars(self.args)
        diff_keys = args_all_dict.keys() - args_original_dict.keys()
        hyperparameter_str = "_".join([f"{i}_{args_all_dict[i]}" for i in diff_keys])
        self.name_4_output_dir = (
            "_".join([hyperparameter_str, begin_time])
            if len(hyperparameter_str) > 0
            else begin_time
        )

        with open(PROJECT_DIR / "data" / self.args.dataset / "args.json", "r") as f:
            self.args.dataset_args = json.load(f)

        # get client party info
        try:
            partition_path = PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")
        self.train_clients: List[int] = partition["separation"]["train"]
        self.test_clients: List[int] = partition["separation"]["test"]
        self.client_num: int = partition["separation"]["total"]

        # init model(s) parameters
        self.device = get_best_device(self.args.use_cuda)

        # get_model_arch() would return a class depends on model's name,
        # then init the model object by indicating the dataset and calling the class.
        # Finally transfer the model object to the target device.
        self.model = get_model_arch(model_name=self.args.model)(
            dataset=self.args.dataset
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
            and self.args.external_model_params_file
            and os.path.isfile(self.args.external_model_params_file)
        ):
            # load pretrained params
            self.global_params_dict = torch.load(
                self.args.external_model_params_file, map_location=self.device
            )
        else:
            self.client_trainable_params = [
                trainable_params(self.model, detach=True) for _ in self.train_clients
            ]

        # system heterogeneity (straggler) setting
        self.clients_local_epoch: List[int] = [self.args.local_epoch] * self.client_num
        if (
            self.args.straggler_ratio > 0
            and self.args.local_epoch > self.args.straggler_min_local_epoch
        ):
            straggler_num = int(self.client_num * self.args.straggler_ratio)
            normal_num = self.client_num - straggler_num
            self.clients_local_epoch = [self.args.local_epoch] * (
                normal_num
            ) + random.choices(
                range(self.args.straggler_min_local_epoch, self.args.local_epoch),
                k=straggler_num,
            )
            random.shuffle(self.clients_local_epoch)

        # To make sure all algorithms run through the same client sampling stream.
        # Some algorithms' implicit operations at client side may disturb the stream if sampling happens at each FL round's beginning.
        self.client_sample_stream = [
            random.sample(
                self.train_clients, max(1, int(self.client_num * self.args.join_ratio))
            )
            for _ in range(self.args.global_epoch)
        ]
        self.selected_clients: List[int] = []
        self.current_epoch = 0
        self.epoch_test = []  # epoch that need test model on test clients.
        # For controlling behaviors of some specific methods while testing (not used by all methods)
        self.test_flag = False

        # variables for logging
        if not os.path.isdir(OUT_DIR / self.algo / self.name_4_output_dir) and (
            self.args.save_log or self.args.save_fig or self.args.save_metrics
        ):
            os.makedirs(OUT_DIR / self.algo / self.name_4_output_dir, exist_ok=True)

        if self.args.visible:
            from visdom import Visdom

            self.viz = Visdom()
            if self.args.viz_win_name is not None:
                self.viz_win_name = self.args.viz_win_name
            else:
                self.viz_win_name = (
                    f"{self.algo}"
                    + f"_{self.args.dataset}"
                    + f"_{self.args.global_epoch}"
                    + f"_{self.args.local_epoch}"
                )
        self.client_stats = {i: {} for i in self.train_clients}
        self.metrics = {
            "train_before": [],
            "train_after": [],
            "test_before": [],
            "test_after": [],
            "test_clients": [],  # test accuracy on test clients
            "mean": [],  # mean accuracy of test_before and test_clients
        }
        stdout = Console(log_path=False, log_time=False)
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.save_log,
            logfile_path=OUT_DIR
            / self.algo
            / self.name_4_output_dir
            / f"{self.args.dataset}_log.html",
        )
        self.test_results: Dict[int, Dict[str, str]] = {}
        self.train_progress_bar = track(
            range(self.args.global_epoch), "[bold green]Training...", console=stdout
        )

        self.logger.log("=" * 20, "ALGORITHM:", self.algo, "=" * 20)
        self.logger.log("Experiment Arguments:", dict(self.args._get_kwargs()))

        # init trainer
        self.trainer = None
        if default_trainer:
            self.trainer = FedAvgClient(
                deepcopy(self.model), self.args, self.logger, self.device
            )

    def train(self):
        """The Generic FL training process"""
        avg_round_time = 0
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E + 1}", "-" * 26)

            if (E + 1) % self.args.test_gap == 0:
                self.epoch_test.append(E)
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
            (
                delta,
                weight,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )

            delta_cache.append(delta)
            weight_cache.append(weight)

        self.aggregate(delta_cache, weight_cache)

    def test(self):
        """The function for testing FL method's output (a single global model or personalized client models)."""
        self.test_flag = True
        loss_before, loss_after = [], []
        correct_before, correct_after = [], []
        num_samples = []
        for client_id in self.test_clients:
            client_local_params = self.generate_client_params(client_id)
            stats = self.trainer.test(client_id, client_local_params)

            correct_before.append(stats["before"]["test_correct"])
            correct_after.append(stats["after"]["test_correct"])
            loss_before.append(stats["before"]["test_loss"])
            loss_after.append(stats["after"]["test_loss"])
            num_samples.append(stats["before"]["test_size"])

        loss_before = torch.tensor(loss_before)
        loss_after = torch.tensor(loss_after)
        correct_before = torch.tensor(correct_before)
        correct_after = torch.tensor(correct_after)
        num_samples = torch.tensor(num_samples)

        self.test_results[self.current_epoch + 1] = {
            "loss": "{:.4f} -> {:.4f}".format(
                loss_before.sum() / num_samples.sum(),
                loss_after.sum() / num_samples.sum(),
            ),
            "accuracy": "{:.2f}% -> {:.2f}%".format(
                correct_before.sum() / num_samples.sum() * 100,
                correct_after.sum() / num_samples.sum() * 100,
            ),
        }
        if len(self.test_clients) > 0:
            self.metrics["test_clients"].append(
                correct_before.sum() / num_samples.sum() * 100
            )
            if (self.current_epoch + 1) % self.args.verbose_gap == 0:
                self.logger.log(
                    f"The test accuracy on test clients before training: {self.metrics['test_clients'][-1]:.2f}%."
                )
        self.test_flag = False

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

    def check_convergence(self):
        """This function is for checking model convergence through the entire FL training process."""
        for label, metric in self.metrics.items():
            if len(metric) > 0:
                self.logger.log(f"Convergence ({label}):")
                acc_range = [90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
                min_acc_idx = 10
                max_acc = 0
                for E, acc in enumerate(metric):
                    for i, target in enumerate(acc_range):
                        if acc >= target and acc > max_acc:
                            self.logger.log(
                                "{} achieved {}%({:.2f}%) at epoch: {}".format(
                                    self.algo, target, acc, E
                                )
                            )
                            max_acc = acc
                            min_acc_idx = i
                            break
                    acc_range = acc_range[:min_acc_idx]

    def log_info(self):
        """This function is for logging each selected client's training info."""
        split = self.args.dataset_args["split"]
        label = {"sample": "test", "user": "train", "domain": "test"}[split]
        correct_before = torch.tensor(
            [
                self.client_stats[i][self.current_epoch]["before"][f"{label}_correct"]
                for i in self.selected_clients
            ]
        )
        correct_after = torch.tensor(
            [
                self.client_stats[i][self.current_epoch]["after"][f"{label}_correct"]
                for i in self.selected_clients
            ]
        )
        num_samples = torch.tensor(
            [
                self.client_stats[i][self.current_epoch]["before"][f"{label}_size"]
                for i in self.selected_clients
            ]
        )

        acc_before = (
            correct_before.sum(dim=-1, keepdim=True) / num_samples.sum() * 100.0
        ).item()
        acc_after = (
            correct_after.sum(dim=-1, keepdim=True) / num_samples.sum() * 100.0
        ).item()
        self.metrics[f"{label}_before"].append(acc_before)
        self.metrics[f"{label}_after"].append(acc_after)
        if (self.current_epoch + 1) % self.args.test_gap == 0 and len(
            self.metrics["test_clients"]
        ) > 0:
            self.metrics["mean"].append(
                (self.metrics["test_before"][-1] + self.metrics["test_clients"][-1]) / 2
            )
            if (self.current_epoch + 1) % self.args.verbose_gap == 0:
                self.logger.log(
                    f"The average test accuracy on training clients {self.metrics['test_before'][-1]:.2f}% -> {self.metrics['test_after'][-1]:.2f}%."
                )
        if self.args.visible:
            self.viz.line(
                [acc_before],
                [self.current_epoch],
                win=self.viz_win_name,
                update="append",
                name=f"{label}(before)",
                opts=dict(
                    title=self.viz_win_name,
                    xlabel="Communication Rounds",
                    ylabel="Accuracy",
                ),
            )
            self.viz.line(
                [acc_after],
                [self.current_epoch],
                win=self.viz_win_name,
                update="append",
                name=f"{label}(after)",
            )

    def log_max_metrics(
        self,
    ):
        self.logger.log("=" * 20, self.algo, "MAX ACC:", "=" * 20)
        # find the max metrics
        max_test_before = max(self.metrics["test_before"])
        max_test_clients = max(self.metrics["test_clients"])
        max_mean = max(self.metrics["mean"])
        # find the index
        max_test_before_index = self.metrics["test_before"].index(max_test_before)
        max_test_clients_index = self.metrics["test_clients"].index(max_test_clients)
        max_mean_index = self.metrics["mean"].index(max_mean)
        # log the max metrics
        self.logger.log(
            f"max_test_before: {max_test_before:.2f}% at epoch: {max_test_before_index+1}"
        )
        self.logger.log(
            f"max_test_clients: {max_test_clients:.2f}% at epoch: {self.epoch_test[max_test_clients_index]+1}"
        )
        self.logger.log(
            f"max_mean: {max_mean:.2f}% at epoch: {self.epoch_test[max_mean_index]+1}. test_before: {self.metrics['test_before'][self.epoch_test[max_mean_index]]:.2f}%, test_clients:{self.metrics['test_clients'][self.epoch_test[max_mean_index]]:.2f}%"
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

        if self.args.visible:
            self.viz.close(win=self.viz_win_name)

        self.train()
        end = time.time()
        total = end - begin
        self.logger.log(
            f"{self.algo}'s total running time: {int(total // 3600)} h {int((total % 3600) // 60)} m {int(total % 60)} s."
        )
        self.logger.log(
            "=" * 20, self.algo, "TEST RESULTS:", "=" * 20, self.test_results
        )

        if self.args.check_convergence:
            self.check_convergence()
        self.log_max_metrics()
        self.logger.close()

        if self.args.save_fig:
            import matplotlib
            from matplotlib import pyplot as plt

            matplotlib.use("Agg")
            linestyle = {
                "test_before": "solid",
                "test_after": "solid",
                "train_before": "dotted",
                "train_after": "dotted",
                "test_clients": "dashed",
                "mean": "dashdot",
            }
            for label, acc in self.metrics.items():
                if len(acc) > 0:
                    # because test_clients and mean are generated according to -tg while other metrics are generated in each epoch.
                    # so we need to plot them in different ways.
                    if label not in ("test_clients", "mean"):
                        plt.plot(acc, label=label, ls=linestyle[label])
                    else:
                        plt.plot(self.epoch_test, acc, label=label, ls=linestyle[label])
            plt.title(f"{self.algo}_{self.args.dataset}")
            plt.ylim(0, 100)
            plt.xlabel("Communication Rounds")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(
                OUT_DIR
                / self.algo
                / self.name_4_output_dir
                / f"{self.args.dataset}.jpeg",
                bbox_inches="tight",
            )
        if self.args.save_metrics:
            import pandas as pd
            import numpy as np

            accuracies = []
            labels = []
            df = pd.DataFrame()
            for label, acc in self.metrics.items():
                if len(acc) > 0:
                    if label not in ("test_clients", "mean"):
                        df.insert(df.shape[1], label, np.array(acc).T)
                    else:
                        df.insert(
                            df.shape[1],
                            label,
                            pd.Series(np.array(acc).T, index=self.epoch_test),
                        )
            df.to_csv(
                OUT_DIR
                / self.algo
                / self.name_4_output_dir
                / f"{self.args.dataset}_acc_metrics.csv",
                index=True,
                index_label="epoch",
            )
        # save trained model(s)
        if self.args.save_model:
            model_name = (
                f"{self.args.dataset}_{self.args.global_epoch}_{self.args.model}.pt"
            )
            if self.unique_model:
                torch.save(
                    self.client_trainable_params,
                    OUT_DIR / self.algo / self.name_4_output_dir / model_name,
                )
            else:
                torch.save(
                    self.global_params_dict,
                    OUT_DIR / self.algo / self.name_4_output_dir / model_name,
                )


if __name__ == "__main__":
    server = FedAvgServer()
    server.run()
