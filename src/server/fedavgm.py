from argparse import ArgumentParser, Namespace
import torch

from fedavg import FedAvgServer
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
        default_trainer=True,
    ):
        super().__init__(args, algo, unique_model, default_trainer)
        self.global_optmizer = torch.optim.SGD(
            list(self.global_params_dict.values()),
            lr=1.0,
            momentum=self.args.fedavgm.server_momentum,
            nesterov=True,
        )

    @torch.no_grad()
    def aggregate(self, delta_cache, weight_cache):
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)

        delta_list = [list(delta.values()) for delta in delta_cache]

        aggregated_delta = []
        for layer_delta in zip(*delta_list):
            aggregated_delta.append(
                torch.sum(torch.stack(layer_delta, dim=-1) * weights, dim=-1)
            )

        self.global_optmizer.zero_grad()
        for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
            param.grad = diff.data
        self.global_optmizer.step()
