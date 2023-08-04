from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.ditto import DittoClient


def get_ditto_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--pers_epoch", type=int, default=1)
    parser.add_argument("--lamda", type=float, default=1)
    return parser


class DittoServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "Ditto",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_ditto_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = DittoClient(
            deepcopy(self.model), self.args, self.logger, self.device, self.client_num
        )


if __name__ == "__main__":
    server = DittoServer()
    server.run()
