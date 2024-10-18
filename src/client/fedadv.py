from copy import deepcopy
from typing import Any
from torch.nn import BatchNorm2d

from src.client.fedavg import FedAvgClient
from src.utils.my_utils import LayerFilter


class FedAdvClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        # self.post_layer_filter = None
        # self.local_round_idx = 0

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.local_round_idx = package['local_round_idx']
        self.post_layer_filter = package['post_layer_filter']

    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        self.train_with_eval()
        self.local_round_idx += 1
        client_package = self.package()
        return client_package

    def package(self):
        """Package data that client needs to transmit to the server.
        You can override this function and add more parameters.

        Returns:
            A dict: {
                `weight`: Client weight. Defaults to the size of client training set.
                `regular_model_params`: Client model parameters that will join parameter aggregation.
                `model_params_diff`: The parameter difference between the client trained and the global. `diff = global - trained`.
                `eval_results`: Client model evaluation results.
                `personal_model_params`: Client model parameters that absent to parameter aggregation.
                `optimzier_state`: Client optimizer's state dict.
                `lr_scheduler_state`: Client learning rate scheduler's state dict.
            }
        """
        regular_params:dict = deepcopy(self.post_layer_filter(self.model.state_dict()))
        for key in regular_params.keys():
            regular_params[key] = regular_params[key].cpu()

        post_personal_filter = LayerFilter(unselect_keys=list(regular_params.keys()))
        
        personal_params = deepcopy(post_personal_filter(self.model.state_dict()))
        for key in personal_params.keys():
            personal_params[key] = personal_params[key].cpu()
        
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            regular_model_params=regular_params,
            personal_model_params=personal_params,
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
            local_round_idx=self.local_round_idx,
        )
        if self.return_diff:
            client_package["model_params_diff"] = {
                key: param_old - param_new
                for (key, param_new), param_old in zip(
                    client_package["regular_model_params"].items(),
                    self.global_regular_model_params.values(),
                )
            }
            client_package.pop("regular_model_params")
        return client_package