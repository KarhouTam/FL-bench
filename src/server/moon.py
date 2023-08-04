from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.moon import MOONClient


def get_moon_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=5)
    return parser


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
        self.trainer = MOONClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )


if __name__ == "__main__":
    server = MOONServer()
    server.run()
