from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_fedprox_argparser
from src.client.fedprox import FedProxClient


class FedProxServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedProx",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedprox_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedProxClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = FedProxServer()
    server.run()
