from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy

import torch
from omegaconf import DictConfig

from src.server.fedavg import FedAvgServer


class FedOptServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument(
            "--type", choices=["adagrad", "yogi", "adam"], type=str, default="adam"
        )
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.999)
        parser.add_argument("--server_lr", type=float, default=1e-1)
        parser.add_argument("--tau", type=float, default=1e-3)
        return parser.parse_args(args_list)

    algo_names = {"adagrad": "FedAdagrad", "yogi": "FedYogi", "adam": "FedAdam"}

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedAvg",
        unique_model=False,
        use_fedavg_client_cls=True,
        return_diff=True,
    ):
        algo = self.algo_names[args.fedopt.type]
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.adaptive_optimizer = AdaptiveOptimizer(
            optimizer_type=self.args.fedopt.type,
            params_dict=self.public_model_params,
            beta1=self.args.fedopt.beta1,
            beta2=self.args.fedopt.beta2,
            lr=self.args.fedopt.server_lr,
            tau=self.args.fedopt.tau,
        )

    def train_one_round(self):
        client_packages = self.trainer.train()
        clients_model_params_diff = []
        client_weights = []
        for package in client_packages.values():
            clients_model_params_diff.append(package["model_params_diff"])
            client_weights.append(package["weight"])

        self.adaptive_optimizer.step(
            clients_model_params_diff=clients_model_params_diff,
            weights=torch.tensor(client_weights) / sum(client_weights),
        )


class AdaptiveOptimizer:
    def __init__(
        self,
        optimizer_type: str,
        params_dict: OrderedDict[str, torch.Tensor],
        beta1: float,
        beta2: float,
        lr: float,
        tau: float,
    ):
        self.update = {
            "adagrad": self._update_adagrad,
            "yogi": self._update_yogi,
            "adam": self._update_adam,
        }[optimizer_type]
        self.params_dict = params_dict
        self.lr = lr
        self.tau = tau
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentums = [
            torch.zeros_like(param) for param in self.params_dict.values()
        ]
        self.velocities = deepcopy(self.momentums)
        self.delta_list: list[torch.Tensor] = None

    @torch.no_grad()
    def step(
        self,
        clients_model_params_diff: list[OrderedDict[str, torch.Tensor]],
        weights: torch.Tensor,
    ):
        # compute weighted delta
        list_clients_model_params_diff = [
            [-diff for diff in diff_dict.values()]
            for diff_dict in clients_model_params_diff
        ]
        params_diff = []
        for diff in zip(*list_clients_model_params_diff):
            params_diff.append(torch.sum(torch.stack(diff, dim=-1) * weights, dim=-1))

        # update momentums
        for m, diff in zip(self.momentums, params_diff):
            m.data = self.beta1 * m + (1 - self.beta1) * diff

        # update velocities according to different rules
        self.update(params_diff)

        # update model parameters
        for param, m, v in zip(
            self.params_dict.values(), self.momentums, self.velocities
        ):
            param.data = param.data + self.lr * (m / (v.sqrt() + self.tau))

    def _update_adagrad(self, delta_list):
        for v, delta in zip(self.velocities, delta_list):
            v.data = v + delta**2

    def _update_yogi(self, delta_list):
        for v, delta in zip(self.velocities, delta_list):
            delta_pow2 = delta**2
            v.data = v - (1 - self.beta2) * delta_pow2 * torch.sign(v - delta_pow2)

    def _update_adam(self, delta_list):
        for v, delta in zip(self.velocities, delta_list):
            v.data = self.beta2 * v + (1 - self.beta2) * delta**2
