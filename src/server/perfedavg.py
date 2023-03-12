from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_perfedavg_argparser
from src.client.perfedavg import PerFedAvgClient


class PerFedAvgServer(FedAvgServer):
    def __init__(
        self,
        algo: str = None,
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_perfedavg_argparser().parse_args()
        algo = "Per-FedAvg(FO)" if args.version == "fo" else "Per-FedAvg(HF)"
        args.finetune_epoch = max(1, args.finetune_epoch)
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = PerFedAvgClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = PerFedAvgServer()
    server.run()
