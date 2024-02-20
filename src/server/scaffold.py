from copy import deepcopy
from typing import List
from argparse import ArgumentParser, Namespace

import torch

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.scaffold import SCAFFOLDClient
from src.utils.tools import trainable_params


def get_scaffold_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--global_lr", type=float, default=1.0)
    return parser


class SCAFFOLDServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "SCAFFOLD",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_scaffold_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
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
            (
                y_delta,
                c_delta,
                self.client_metrics[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                c_global=self.c_global,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
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
                self.args.global_lr * torch.stack(y_delta, dim=-1).mean(dim=-1)
            )

        # update global control
        for c_global, c_delta in zip(self.c_global, zip(*c_delta_cache)):
            c_global.data.add_(
                torch.stack(c_delta, dim=-1).sum(dim=-1) / self.client_num
            )


if __name__ == "__main__":
    server = SCAFFOLDServer()
    server.run()
