from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.client.fedrep import FedRepClient
from src.utils.tools import NestedNamespace


def get_fedrep_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--train_body_epoch", type=int, default=1)
    return parser.parse_args(args_list)


class FedRepServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedRep",
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(args, algo, unique_model, default_trainer)
        self.trainer = FedRepClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
