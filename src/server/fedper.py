from copy import deepcopy

from fedavg import FedAvgServer
from src.client.fedper import FedPerClient
from src.utils.tools import NestedNamespace


class FedPerServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedPer",
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(args, algo, unique_model, default_trainer)
        self.trainer = FedPerClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
