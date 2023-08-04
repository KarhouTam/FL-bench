from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.fedprox import FedProxClient


def get_fedprox_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--mu", type=float, default=1.0)
    return parser


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
        self.trainer = FedProxClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )


if __name__ == "__main__":
    server = FedProxServer()
    server.run()
