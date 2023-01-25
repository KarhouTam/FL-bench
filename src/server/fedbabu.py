from copy import deepcopy

from fedavg import FedAvgServer
from config.args import get_fedavg_argparser
from client.fedbabu import FedBabuClient


class FedBABUServer(FedAvgServer):
    def __init__(self):
        args = get_fedavg_argparser().parse_args()
        args.finetune_epoch = max(1, args.finetune_epoch)
        super().__init__("FedBABU", args=args, default_trainer=False)
        self.trainer = FedBabuClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = FedBABUServer()
    server.run()
