from copy import deepcopy

from fedavg import FedAvgServer
from client.fedbn import FedBNClient


class FedBNServer(FedAvgServer):
    def __init__(self):
        super().__init__("FedBN", default_trainer=False)
        self.trainer = FedBNClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = FedBNServer()
    server.run()
