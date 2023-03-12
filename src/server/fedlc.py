from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_fedlc_argparser
from src.client.fedlc import FedLCClient


class FedLCServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedLC",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedlc_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedLCClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = FedLCServer()
    server.run()
