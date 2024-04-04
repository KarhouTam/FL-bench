from copy import deepcopy

from fedavg import FedAvgServer
from src.client.fedbabu import FedBabuClient
from src.utils.tools import NestedNamespace


class FedBabuServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedBabu",
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(args, algo, unique_model, default_trainer)
        # Fine-tuning is indispensable to FedBabu.
        self.args.common.finetune_epoch = max(1, self.args.common.finetune_epoch)
        self.trainer = FedBabuClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
