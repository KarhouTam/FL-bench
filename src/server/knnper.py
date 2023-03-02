from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_knnper_argparser
from src.client.knnper import kNNPerClient


class kNNPerServer(FedAvgServer):
    def __init__(self):
        args = get_knnper_argparser().parse_args()
        super().__init__("kNN-Per", args, default_trainer=False)
        self.trainer = kNNPerClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = kNNPerServer()
    server.run()
