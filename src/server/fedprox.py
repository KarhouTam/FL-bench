from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.client.fedprox import FedProxClient
from src.utils.tools import NestedNamespace


def get_fedprox_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--mu", type=float, default=1.0)
    return parser.parse_args(args_list)


class FedProxServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedProx",
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(args, algo, unique_model, default_trainer)
        self.trainer = FedProxClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
