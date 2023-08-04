from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.client.fedbabu import FedBabuClient


class FedBabuServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedBabu",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(algo, args, unique_model, default_trainer)
        # Fine-tuning is indispensable to FedBabu.
        self.args.finetune_epoch = max(1, self.args.finetune_epoch)
        self.trainer = FedBabuClient(deepcopy(self.model), self.args, self.logger, self.device)


if __name__ == "__main__":
    server = FedBabuServer()
    server.run()
