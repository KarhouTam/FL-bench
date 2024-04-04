from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.client.apfl import APFLClient
from src.utils.tools import NestedNamespace


def get_apfl_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--adaptive_alpha", type=int, default=1)
    return parser.parse_args(args_list)


class APFLServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "APFL",
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(args, algo, unique_model, default_trainer)
        self.trainer = APFLClient(
            deepcopy(self.model), self.args, self.logger, self.device, self.client_num
        )
