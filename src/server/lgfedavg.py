from copy import deepcopy

from fedavg import FedAvgServer
from config.args import get_lgfedavg_argparser
from client.lgfedavg import LG_FedAvgClient


class LG_FedAvgServer(FedAvgServer):
    def __init__(self):
        args = get_lgfedavg_argparser().parse_args()
        super().__init__("LG-FedAvg", args=args, default_trainer=False)
        self.trainer = LG_FedAvgClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = LG_FedAvgServer()
    server.run()
