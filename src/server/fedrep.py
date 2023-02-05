from copy import deepcopy

from fedavg import FedAvgServer
from config.args import get_fedrep_argparser
from client.fedrep import FedRepClient


class FedRepServer(FedAvgServer):
    def __init__(self):
        args = get_fedrep_argparser().parse_args()
        args.finetune_epoch = max(1, args.finetune_epoch)
        super().__init__("FedRep", args, default_trainer=False)
        self.trainer = FedRepClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = FedRepServer()
    server.run()
