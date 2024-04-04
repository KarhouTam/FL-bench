from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.client.ditto import DittoClient
from src.utils.tools import NestedNamespace


def get_ditto_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--pers_epoch", type=int, default=1)
    parser.add_argument("--lamda", type=float, default=1)
    return parser.parse_args(args_list)


class DittoServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "Ditto",
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(args, algo, unique_model, default_trainer)
        self.trainer = DittoClient(
            deepcopy(self.model), self.args, self.logger, self.device, self.client_num
        )
