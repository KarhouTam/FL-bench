from argparse import ArgumentParser, Namespace
from copy import deepcopy

import torch

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.feddyn import FedDynClient
from src.config.utils import trainable_params


def get_feddyn_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--alpha", type=float, default=0.01)
    return parser


class FedDynServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedDyn",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_feddyn_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedDynClient(deepcopy(self.model), self.args, self.logger)
        self.h = [
            torch.zeros_like(param, device=self.device)
            for param in trainable_params(self.model)
        ]

    @torch.no_grad()
    def aggregate(self, client_params_cache, weight_cache):
        aggregated_delta = [
            torch.sum(torch.stack(client_params, dim=0) - global_param, dim=0)
            for client_params, global_param in zip(
                zip(*client_params_cache), self.global_params_dict.values()
            )
        ]
        self.h = [
            prev_h - (self.args.alpha / self.client_num) * delta
            for prev_h, delta in zip(self.h, aggregated_delta)
        ]

        new_parameters = [
            torch.stack(client_param, dim=0).mean(dim=0) - h / self.args.alpha
            for client_param, h in zip(zip(*client_params_cache), self.h)
        ]

        for param_old, param_new in zip(
            self.global_params_dict.values(), new_parameters
        ):
            param_old.data = param_new.data


if __name__ == '__main__':
    server = FedDynServer()
    server.run()
