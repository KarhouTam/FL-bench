from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.client.moon import MOONClient
from src.utils.tools import NestedNamespace


def get_moon_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=5)
    return parser.parse_args(args_list)


class MOONServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "MOON",
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(args, algo, unique_model, default_trainer)
        self.trainer = MOONClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
