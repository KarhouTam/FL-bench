from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_lgfedavg_argparser
from src.client.lgfedavg import LGFedAvgClient


class LGFedAvgServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "LG-FedAvg",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_lgfedavg_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = LGFedAvgClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = LGFedAvgServer()
    server.run()
