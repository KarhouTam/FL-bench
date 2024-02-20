from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import List, OrderedDict

import torch

from torch._tensor import Tensor
from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.elastic import ElasticClient


def get_elastic_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.95)
    return parser


class ElasticServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "Elastic",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = ElasticClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
        self.client_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.trainer.optimizer, self.args.global_epoch, verbose=False
        )

    def train_one_round(self):
        delta_cache = []
        weight_cache = []
        sensitivity_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            (
                delta,
                weight,
                self.client_metrics[client_id][self.current_epoch],
                sensitivity,
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                return_diff=True,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )

            delta_cache.append(delta)
            weight_cache.append(weight)
            sensitivity_cache.append(sensitivity)

        self.aggregate(delta_cache, weight_cache, sensitivity_cache)
        self.client_lr_scheduler.step()

    def aggregate(
        self,
        delta_cache: List[OrderedDict[str, Tensor]],
        weight_cache: List[int],
        sensitivity_cache: List[torch.Tensor],
    ):
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        stacked_sensitivity = torch.stack(sensitivity_cache, dim=-1)
        aggregated_sensitivity = torch.sum(stacked_sensitivity * weights, dim=-1)
        max_sensitivity = stacked_sensitivity.max(dim=-1)[0]
        zeta = 1 + self.args.tau - aggregated_sensitivity / max_sensitivity
        delta_list = [list(delta.values()) for delta in delta_cache]
        aggregated_delta = [
            torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
            for diff in zip(*delta_list)
        ]

        for param, coef, diff in zip(
            self.global_params_dict.values(), zeta, aggregated_delta
        ):
            param.data -= coef * diff
        self.model.load_state_dict(self.global_params_dict, strict=False)
