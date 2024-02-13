from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import List, OrderedDict

import torch
import torch.nn as nn

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.fedrod import FedRoDClient
from src.utils.tools import trainable_params
from src.utils.constants import NUM_CLASSES


def get_fedrod_argparser():
    parser = get_fedavg_argparser()
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--hyper", type=int, default=0)
    parser.add_argument("--hyper_lr", type=float, default=0.1)
    parser.add_argument("--hyper_hidden_dim", type=int, default=32)
    parser.add_argument("--eval_per", type=int, default=1)
    return parser


class FedRoDServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedRoD",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(algo, args, unique_model, default_trainer)
        self.hyper_params_dict = None
        self.hypernetwork: nn.Module = None
        if self.args.hyper:
            output_dim = (
                self.model.classifier.weight.numel()
                + self.model.classifier.bias.numel()
            )
            input_dim = NUM_CLASSES[self.args.dataset]
            self.hypernetwork = HyperNetwork(
                input_dim, self.args.hyper_hidden_dim, output_dim
            ).to(self.device)
            params, keys = trainable_params(
                self.hypernetwork, detach=True, requires_name=True
            )
            self.hyper_params_dict = OrderedDict(zip(keys, params))
        self.trainer = FedRoDClient(
            model=deepcopy(self.model),
            hypernetwork=deepcopy(self.hypernetwork),
            args=self.args,
            logger=self.logger,
            device=self.device,
        )

    def train_one_round(self):
        delta_cache = []
        weight_cache = []
        hyper_delta_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            (
                delta,
                hyper_delta,
                weight,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                hyper_parameters=self.hyper_params_dict,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
                return_diff=False,
            )

            delta_cache.append(delta)
            weight_cache.append(weight)
            hyper_delta_cache.append(hyper_delta)

        self.aggregate(delta_cache, hyper_delta_cache, weight_cache)

    @torch.no_grad()
    def aggregate(
        self,
        delta_cache: List[OrderedDict[str, torch.Tensor]],
        hyper_delta_cache: List[OrderedDict[str, torch.Tensor]],
        weight_cache: List[int],
        return_diff=False,
    ):
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        if return_diff:
            delta_list = [list(delta.values()) for delta in delta_cache]
            aggregated_delta = [
                torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
                for diff in zip(*delta_list)
            ]
            for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
                param.data -= diff

            if self.args.hyper:
                hyper_delta_list = [list(delta.values()) for delta in delta_cache]
                aggregated_hyper_delta = [
                    torch.sum(weights * torch.stack(diff, dim=-1), dim=-1)
                    for diff in zip(*hyper_delta_list)
                ]
                for param, diff in zip(
                    self.hyper_params_dict.values(), aggregated_hyper_delta
                ):
                    param.data -= diff

        else:
            for old_param, zipped_new_param in zip(
                self.global_params_dict.values(), zip(*delta_cache)
            ):
                old_param.data = (torch.stack(zipped_new_param, dim=-1) * weights).sum(
                    dim=-1
                )
            self.model.load_state_dict(self.global_params_dict, strict=False)

            if self.args.hyper:
                for old_param, zipped_new_param in zip(
                    self.hyper_params_dict.values(), zip(*hyper_delta_cache)
                ):
                    old_param.data = (
                        torch.stack(zipped_new_param, dim=-1) * weights
                    ).sum(dim=-1)

        if self.args.hyper:
            self.hypernetwork.load_state_dict(self.hyper_params_dict)
        self.model.load_state_dict(self.global_params_dict, strict=False)


class HyperNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)


