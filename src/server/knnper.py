from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.knnper import kNNPerClient


def get_knnper_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--capacity", type=int, default=500)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--k", type=int, default=5)
    return parser


class kNNPerServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "kNN-Per",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_knnper_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = kNNPerClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )


if __name__ == "__main__":
    server = kNNPerServer()
    server.run()
