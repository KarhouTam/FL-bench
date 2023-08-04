from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.client.fedper import FedPerClient


class FedPerServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedPer",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedPerClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )


if __name__ == "__main__":
    server = FedPerServer()
    server.run()
