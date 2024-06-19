from argparse import ArgumentParser, Namespace
from typing import Any, OrderedDict
import torch

from src.server.fedavg import FedAvgServer
from src.utils.tools import NestedNamespace


class FedAvgMServer(FedAvgServer):

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--server_momentum", type=float, default=0.9)
        return parser.parse_args(args_list)

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
            list(self.public_model_params.values()),
            lr=1.0,
            momentum=self.args.fedavgm.server_momentum,
            nesterov=True,
        )

    @torch.no_grad()
    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):
        self.global_optmizer.zero_grad()

        clients_weight = [package["weight"] for package in clients_package.values()]
        weights = torch.tensor(clients_weight) / sum(clients_weight)
        for key, global_param in self.public_model_params.items():
            if "num_batches_tracked" not in key:
                diffs = torch.stack(
                    [
                        package["model_params_diff"][key]
                        for package in clients_package.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(diffs * weights, dim=-1).to(global_param.device)
                self.public_model_params[key].grad = aggregated

        self.global_optmizer.step()
