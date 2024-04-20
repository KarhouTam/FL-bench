from argparse import ArgumentParser, Namespace
from typing import Any, OrderedDict
import torch

from src.server.fedavg import FedAvgServer
from src.utils.tools import NestedNamespace


def get_fedavgm_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--server_momentum", type=float, default=0.9)
    return parser.parse_args(args_list)


class FedAvgMServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedAvgM",
        unique_model=False,
        use_fedavg_client_cls=True,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.global_optmizer = torch.optim.SGD(
            list(self.global_model_params.values()),
            lr=1.0,
            momentum=self.args.fedavgm.server_momentum,
            nesterov=True,
        )

    @torch.no_grad()
    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):
        clients_weight = [package["weight"] for package in clients_package.values()]
        weights = torch.tensor(clients_weight) / sum(clients_weight)
        params_diff = [
            list(package["model_params_diff"].values())
            for package in clients_package.values()
        ]

        aggregated_diff = []
        for layer_diff in zip(*params_diff):
            aggregated_diff.append(
                torch.sum(torch.stack(layer_diff, dim=-1) * weights, dim=-1)
            )

        self.global_optmizer.zero_grad()
        for param, diff in zip(self.global_model_params.values(), aggregated_diff):
            param.grad = diff.data
        self.global_optmizer.step()
