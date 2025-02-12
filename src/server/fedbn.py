from copy import deepcopy

from omegaconf import DictConfig

from src.client.fedbn import FedBNClient
from src.server.fedavg import FedAvgServer


class FedBNServer(FedAvgServer):
    algorithm_name = "FedBN"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedBNClient

    def __init__(self, args: DictConfig):
        super().__init__(args)
        init_bn_params = dict(
            (key, param)
            for key, param in self.model.state_dict().items()
            if "bn" in key
        )
        for params_dict in self.clients_personal_model_params.values():
            params_dict.update(deepcopy(init_bn_params))
