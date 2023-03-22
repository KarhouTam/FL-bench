from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_fedavg_argparser
from src.client.fedbabu import FedBabuClient


class FedBabuServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedBabu",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedavg_argparser().parse_args()
        args.finetune_epoch = max(1, args.finetune_epoch)
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedBabuClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = FedBabuServer()
    server.run()
