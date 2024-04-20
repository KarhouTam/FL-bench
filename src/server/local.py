from src.server.fedavg import FedAvgServer
from src.utils.tools import NestedNamespace


class LocalServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "Local-only",
        unique_model=True,
        use_fedavg_client_cls=True,
        return_diff=False,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
