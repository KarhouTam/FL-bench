from copy import deepcopy

from fedavg import FedAvgServer
from src.client.fedper import FedPerClient
from src.config.args import get_fedavg_argparser


class FedPerServer(FedAvgServer):
    def __init__(self):
        args = get_fedavg_argparser().parse_args()
        args.finetune_epoch = max(1, args.finetune_epoch)
        super().__init__("FedPer", args, default_trainer=False)
        self.trainer = FedPerClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = FedPerServer()
    server.run()
