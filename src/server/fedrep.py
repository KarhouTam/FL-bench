from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_fedrep_argparser
from src.client.fedrep import FedRepClient


class FedRepServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedRep",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedrep_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedRepClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = FedRepServer()
    server.run()
