from copy import deepcopy

from fedavg import FedAvgServer
from config.args import get_apfl_argparser
from client.apfl import APFLClient


class APFLServer(FedAvgServer):
    def __init__(self):
        super().__init__(
            "APFL",
            get_apfl_argparser().parse_args(),
            unique_model=False,
            default_trainer=False,
        )
        self.trainer = APFLClient(
            deepcopy(self.model), self.args, self.logger, self.client_num_in_total
        )


if __name__ == "__main__":
    server = APFLServer()
    server.run()
