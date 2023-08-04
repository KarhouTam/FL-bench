from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.client.fedbn import FedBNClient


class FedBNServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedBN",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedBNClient(deepcopy(self.model), self.args, self.logger, self.device)


if __name__ == "__main__":
    server = FedBNServer()
    server.run()
