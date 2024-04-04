from copy import deepcopy
from typing import List
from argparse import ArgumentParser, Namespace

import torch

from fedavg import FedAvgServer
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
        default_trainer=False,
    ):
        super().__init__(args, algo, unique_model, default_trainer)
        self.trainer = SCAFFOLDClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
        self.c_global = [
            torch.zeros_like(param) for param in trainable_params(self.model)
        ]

    def train_one_round(self):
        y_delta_cache = []
        c_delta_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            (y_delta, c_delta, self.client_metrics[client_id][self.current_epoch]) = (
                self.trainer.train(
                    client_id=client_id,
                    local_epoch=self.clients_local_epoch[client_id],
                    new_parameters=client_local_params,
                    c_global=self.c_global,
                    verbose=((self.current_epoch + 1) % self.args.common.verbose_gap)
                    == 0,
                )
            )

            y_delta_cache.append(y_delta)
            c_delta_cache.append(c_delta)

        self.aggregate(y_delta_cache, c_delta_cache)

    @torch.no_grad()
    def aggregate(
        self,
        y_delta_cache: List[List[torch.Tensor]],
        c_delta_cache: List[List[torch.Tensor]],
    ):
        for param, y_delta in zip(
            self.global_params_dict.values(), zip(*y_delta_cache)
        ):
            param.data.add_(
                self.args.scaffold.global_lr * torch.stack(y_delta, dim=-1).mean(dim=-1)
            )

        # update global control
        for c_global, c_delta in zip(self.c_global, zip(*c_delta_cache)):
            c_global.data.add_(
                torch.stack(c_delta, dim=-1).sum(dim=-1) / self.client_num
            )
