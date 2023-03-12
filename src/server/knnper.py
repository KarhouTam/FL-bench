from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_knnper_argparser
from src.client.knnper import kNNPerClient


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
        self.trainer = kNNPerClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = kNNPerServer()
    server.run()
