from copy import deepcopy
from typing import Any
from argparse import ArgumentParser, Namespace

import torch

from src.server.fedavg import FedAvgServer
from src.client.scaffold import SCAFFOLDClient
from src.utils.tools import trainable_params, NestedNamespace


def get_scaffold_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--global_lr", type=float, default=1.0)
    return parser.parse_args(args_list)


class SCAFFOLDServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "SCAFFOLD",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.c_global = [
            torch.zeros_like(param) for param in trainable_params(self.model)
        ]
        self.c_local = [deepcopy(self.c_global) for _ in self.train_clients]
        self.init_trainer(SCAFFOLDClient)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["c_global"] = self.c_global
        server_package["c_local"] = self.c_local[client_id]
        return server_package

    @torch.no_grad()
    def aggregate(self, clients_package: dict[int, dict[str, Any]]):
        c_delta_list = [package["c_delta"] for package in clients_package.values()]
        y_delta_list = [package["y_delta"] for package in clients_package.values()]
        for param, y_delta in zip(
            self.global_model_params.values(), zip(*y_delta_list)
        ):
            param.data.add_(
                self.args.scaffold.global_lr * torch.stack(y_delta, dim=-1).mean(dim=-1)
            )

        # update global control
        for c_global, c_delta in zip(self.c_global, zip(*c_delta_list)):
            c_global.data.add_(
                torch.stack(c_delta, dim=-1).sum(dim=-1) / self.client_num
            )
