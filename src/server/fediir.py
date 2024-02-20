from argparse import ArgumentParser, Namespace
from copy import deepcopy

import numpy as np
import torch

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.fediir import FedIIRClient


def get_fediir_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument('--ema', type=float, default=0.95)
    parser.add_argument('--penalty', type=float, default=1e-3)
    return parser


class FedIIRServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedIIR",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_fediir_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedIIRClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
        self.grad_mean = tuple(
            torch.zeros_like(p).to(self.device)
            for p in list(self.model.classifier.parameters())
        )

    def train_one_round(self):
        self.grad_mean = self.calculate_grad_mean()

        delta_cache = []
        weight_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            self.trainer.grad_mean = self.grad_mean
            (
                delta,
                _,
                self.client_metrics[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )

            delta_cache.append(delta)

        weight_cache = list(
            np.ones(len(self.selected_clients)) * (1 / len(self.selected_clients))
        )
        self.aggregate(delta_cache, weight_cache)

    def calculate_grad_mean(self):
        batch_total = 0
        grad_sum = tuple(
            torch.zeros_like(p).to(self.device)
            for p in list(self.model.classifier.parameters())
        )
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            grad_sum_cache, batch_num = self.trainer.grad(
                client_id, client_local_params
            )
            batch_total += batch_num
            grad_sum = tuple(g1 + g2 for g1, g2 in zip(grad_sum, grad_sum_cache))
        grad_mean_new = tuple(grad / batch_total for grad in grad_sum)
        return tuple(
            self.args.ema * g1 + (1 - self.args.ema) * g2
            for g1, g2 in zip(self.grad_mean, grad_mean_new)
        )


if __name__ == "__main__":
    server = FedIIRServer()
    server.run()
