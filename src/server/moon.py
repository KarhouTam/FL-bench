from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_moon_argparser
from src.client.moon import MOONClient


class MOONServer(FedAvgServer):
    def __init__(self):
        super().__init__(
            "MOON", get_moon_argparser().parse_args(), default_trainer=False
        )
        self.trainer = MOONClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = MOONServer()
    server.run()
