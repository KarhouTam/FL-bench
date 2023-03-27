from copy import deepcopy
from typing import List
from argparse import Namespace

import torch

from fedavg import FedAvgServer
from src.client.scaffold import SCAFFOLDClient
from src.config.args import get_scaffold_argparser
from src.config.utils import trainable_params


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
        self.trainer = SCAFFOLDClient(deepcopy(self.model), self.args, self.logger)
        self.c_global = [
            torch.zeros_like(param) for param in trainable_params(self.model)
        ]

    def train(self):
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]

            y_delta_cache = []
            c_delta_cache = []
            for client_id in self.selected_clients:

                client_local_params = self.generate_client_params(client_id)

                (
                    y_delta,
                    c_delta,
                    self.client_stats[client_id][E],
                ) = self.trainer.train(
                    client_id=client_id,
                    new_parameters=client_local_params,
                    c_global=self.c_global,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )

                y_delta_cache.append(y_delta)
                c_delta_cache.append(c_delta)

            self.aggregate(y_delta_cache, c_delta_cache)

            self.log_info()

    @torch.no_grad()
    def aggregate(
        self,
        y_delta_cache: List[List[torch.Tensor]],
        c_delta_cache: List[List[torch.Tensor]],
    ):
        for param, y_delta in zip(
            trainable_params(self.global_params_dict), zip(*y_delta_cache)
        ):
            x_delta = torch.stack(y_delta, dim=-1).mean(dim=-1).to(self.device)
            param.data += self.args.global_lr * x_delta.to(self.device)

        # update global control
        for c_global, c_delta in zip(self.c_global, zip(*c_delta_cache)):
            c_delta = torch.stack(c_delta, dim=-1).sum(dim=-1).to(self.device)
            c_global.data += (1 / self.client_num_in_total) * c_delta.data


if __name__ == "__main__":
    server = SCAFFOLDServer()
    server.run()
