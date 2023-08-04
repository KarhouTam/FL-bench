from argparse import ArgumentParser, Namespace
from copy import deepcopy

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.fedrep import FedRepClient


def get_fedrep_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--train_body_epoch", type=int, default=1)
    return parser


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
        self.trainer = FedRepClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )


if __name__ == "__main__":
    server = FedRepServer()
    server.run()
