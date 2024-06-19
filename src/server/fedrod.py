from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from src.server.fedavg import FedAvgServer
from src.client.fedrod import FedRoDClient
from src.utils.tools import NestedNamespace
from src.utils.constants import NUM_CLASSES


class FedRoDServer(FedAvgServer):

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--gamma", type=float, default=1)
        parser.add_argument("--hyper", type=int, default=0)
        parser.add_argument("--hyper_lr", type=float, default=0.1)
        parser.add_argument("--hyper_hidden_dim", type=int, default=32)
        parser.add_argument("--eval_per", type=int, default=1)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedRoD",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.hyper_params_dict = None
        self.hypernetwork: nn.Module = None
        if self.args.fedrod.hyper:
            output_dim = (
                self.model.classifier.weight.numel()
                + self.model.classifier.bias.numel()
            )
            input_dim = NUM_CLASSES[self.args.common.dataset]
            self.hypernetwork = HyperNetwork(
                input_dim, self.args.fedrod.hyper_hidden_dim, output_dim
            )
            self.hyper_params_dict = OrderedDict()
            for key, param in self.hypernetwork.named_parameters():
                self.hyper_params_dict[key] = param.data.clone()
        self.first_time_selected = [True for _ in self.train_clients]
        self.init_trainer(FedRoDClient, hypernetwork=self.hypernetwork)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["first_time_selected"] = self.first_time_selected[client_id]
        server_package["hypernet_params"] = self.hyper_params_dict
        return server_package

    @torch.no_grad()
    def aggregate(self, clients_package: dict[int, dict[str, Any]]):
        for client_id in clients_package.keys():
            self.first_time_selected[client_id] = False
        clients_weight = [package["weight"] for package in clients_package.values()]
        weights = torch.tensor(clients_weight) / sum(clients_weight)
        clients_model_params_list = [
            list(package["regular_model_params"].values())
            for package in clients_package.values()
        ]
        for old_param, zipped_new_param in zip(
            self.public_model_params.values(), zip(*clients_model_params_list)
        ):
            old_param.data = (torch.stack(zipped_new_param, dim=-1) * weights).sum(
                dim=-1
            )

        if self.args.fedrod.hyper:
            clients_hypernet_params_list = [
                list(package["hypernet_params"].values())
                for package in clients_package.values()
            ]
            for old_param, zipped_new_param in zip(
                self.hyper_params_dict.values(), zip(*clients_hypernet_params_list)
            ):
                old_param.data = (torch.stack(zipped_new_param, dim=-1) * weights).sum(
                    dim=-1
                )


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
