from argparse import Namespace
from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_apfl_argparser
from src.client.apfl import APFLClient


class APFLServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "APFL",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_apfl_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = APFLClient(
            deepcopy(self.model), self.args, self.logger, self.client_num_in_total
        )


if __name__ == "__main__":
    server = APFLServer()
    server.run()
