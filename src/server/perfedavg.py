from copy import deepcopy

from fedavg import FedAvgServer
from src.config.args import get_perfedavg_argparser
from src.client.perfedavg import PerFedAvgClient


class PerFedAvgServer(FedAvgServer):
    def __init__(self):
        args = get_perfedavg_argparser().parse_args()
        args.finetune_epoch = max(1, args.finetune_epoch)
        super().__init__(
            "Per-FedAvg (FO)" if args.version == "fo" else "Per-FedAvg (HF)",
            args=args,
            default_trainer=False,
        )
        self.trainer = PerFedAvgClient(deepcopy(self.model), self.args, self.logger)


if __name__ == "__main__":
    server = PerFedAvgServer()
    server.run()
