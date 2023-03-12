from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_moon_argparser
from src.client.moon import MOONClient


class MOONServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "MOON",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_moon_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = MOONClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = MOONServer()
    server.run()
