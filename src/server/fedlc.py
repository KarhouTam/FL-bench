from copy import deepcopy

from fedavg import FedAvgServer
from config.args import get_fedlc_argparser
from client.fedlc import FedLCClient


class FedLCServer(FedAvgServer):
    def __init__(self):
        super().__init__(
            "FedLC", args=get_fedlc_argparser().parse_args(), default_trainer=False
        )
        self.trainer = FedLCClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = FedLCServer()
    server.run()
