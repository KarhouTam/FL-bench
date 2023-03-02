from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_fedprox_argparser
from src.client.fedprox import FedProxClient


class FedProxServer(FedAvgServer):
    def __init__(self):
        super().__init__(
            "FedProx", args=get_fedprox_argparser().parse_args(), default_trainer=False
        )
        self.trainer = FedProxClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = FedProxServer()
    server.run()
