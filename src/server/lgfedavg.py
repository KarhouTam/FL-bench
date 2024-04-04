from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.client.lgfedavg import LGFedAvgClient
from src.utils.tools import NestedNamespace


def get_lgfedavg_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--num_global_layers", type=int, default=1)
    return parser.parse_args(args_list)


class LGFedAvgServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "LG-FedAvg",
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(args, algo, unique_model, default_trainer)
        self.trainer = LGFedAvgClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
