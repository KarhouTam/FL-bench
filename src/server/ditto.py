from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_ditto_argparser
from src.client.ditto import DittoClient


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
            deepcopy(self.model), self.args, self.logger, self.client_num_in_total
        )


if __name__ == "__main__":
    server = DittoServer()
    server.run()
