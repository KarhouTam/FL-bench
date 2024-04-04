from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import List
from collections import OrderedDict

import torch

from fedavg import FedAvgServer
from src.utils.tools import trainable_params, NestedNamespace
from src.utils.models import DecoupledModel


def get_fedopt_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--type", choices=["adagrad", "yogi", "adam"], type=str, default="adam"
    )
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--server_lr", type=float, default=1e-1)
    parser.add_argument("--tau", type=float, default=1e-3)
    return parser.parse_args(args_list)


ALGO_NAMES = {"adagrad": "FedAdagrad", "yogi": "FedYogi", "adam": "FedAdam"}


class FedOptServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedAvg",
        unique_model=False,
        default_trainer=True,
    ):
        algo = ALGO_NAMES[args.fedopt.type]
        super().__init__(args, algo, unique_model, default_trainer)
        self.adaptive_optimizer = AdaptiveOptimizer(
            optimizer_type=self.args.fedopt.type,
            model=self.model,
            beta1=self.args.fedopt.beta1,
            beta2=self.args.fedopt.beta2,
            lr=self.args.fedopt.server_lr,
            tau=self.args.fedopt.tau,
        )

    def train_one_round(self):
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

        self.adaptive_optimizer.step(
            delta_cache=delta_cache,
            weights=torch.tensor(weight_cache, device=self.device) / sum(weight_cache),
        )

        self.global_params_dict = OrderedDict(
            zip(self.trainable_params_name, trainable_params(self.model, detach=True))
        )


class AdaptiveOptimizer:
    def __init__(
        self,
        optimizer_type: str,
        model: DecoupledModel,
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
        self.model = model
        self.lr = lr
        self.tau = tau
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentums = [
            torch.zeros_like(param) for param in trainable_params(self.model)
        ]
        self.velocities = deepcopy(self.momentums)
        self.delta_list: List[torch.Tensor] = None

    @torch.no_grad()
    def step(
        self, delta_cache: List[OrderedDict[str, torch.Tensor]], weights: torch.Tensor
    ):
        # compute weighted delta
        list_delta_cache = [
            [-diff for diff in delta_dict.values()] for delta_dict in delta_cache
        ]
        delta_list = []
        for delta in zip(*list_delta_cache):
            delta_list.append(torch.sum(torch.stack(delta, dim=-1) * weights, dim=-1))

        # update momentums
        for m, delta in zip(self.momentums, delta_list):
            m.data = self.beta1 * m + (1 - self.beta1) * delta

        # update velocities according to different rules
        self.update(delta_list)

        # update model parameters
        for param, m, v in zip(
            trainable_params(self.model), self.momentums, self.velocities
        ):
            param.data = param.data - self.lr * (m / (v.sqrt() + self.tau))

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
