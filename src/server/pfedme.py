from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import List

import torch

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.pfedme import pFedMeClient


def get_pfedme_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lamda", type=float, default=15)
    parser.add_argument("--pers_lr", type=float, default=0.01)
    parser.add_argument("--mu", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=5)
    return parser


class pFedMeServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "pFedMe",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_pfedme_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = pFedMeClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )

    @torch.no_grad()
    def aggregate(
        self, local_params_cache: List[List[torch.Tensor]], weight_cache: List[int]
    ):
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        aggregated_params = [
            torch.sum(weights * torch.stack(params, dim=-1), dim=-1)
            for params in zip(*local_params_cache)
        ]
        for param_prev, param_new in zip(
            self.global_params_dict.values(), aggregated_params
        ):
            param_prev.data = (
                1 - self.args.beta
            ) * param_prev + self.args.beta * param_new


if __name__ == "__main__":
    server = pFedMeServer()
    server.run()
