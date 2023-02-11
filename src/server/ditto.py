from copy import deepcopy

from fedavg import FedAvgServer
from config.args import get_ditto_argparser
from client.ditto import DittoClient


class DittoServer(FedAvgServer):
    def __init__(self):
        args = get_ditto_argparser().parse_args()
        super().__init__("Ditto", args, default_trainer=False)
        self.trainer = DittoClient(
            deepcopy(self.model), self.args, self.logger, self.client_num_in_total
        )


if __name__ == "__main__":
    server = DittoServer()
    server.run()
