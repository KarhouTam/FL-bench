from copy import deepcopy

from fedavg import FedAvgServer
from src.client.fedbn import FedBNClient
from src.utils.tools import NestedNamespace


class FedBNServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedBN",
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(args, algo, unique_model, default_trainer)
        self.trainer = FedBNClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )
