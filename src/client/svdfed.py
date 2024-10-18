from copy import deepcopy
from typing import Any
import torch

from src.client.fedavg import FedAvgClient
from src.utils.compressor_utils import CompressorCombin

class SVDFedClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.compress_combin:CompressorCombin = None
        # self.combine_error:dict = None

    @torch.no_grad()
    def set_parameters(self, package: dict[str, Any]):
        self.compress_combin = package['compress_combin']
        # self.combine_error = package['combin_error']
        self.full_grad = package['full_grad']
        self.compress_combin.update(package['combin_update_dict'])

        return super().set_parameters(package)


    def pack_client_model(self, raw_model:dict[str, torch.Tensor] , global_model:dict[str, torch.Tensor]):
        grads = {}
        for key in raw_model.keys():
            # quantized_model[key], scale[key] = (global_model[key] - raw_model[key]), 0
            grads[key] = global_model[key] - raw_model[key]

        if self.full_grad:
            return grads
        else:
            combin_alpha, _, combin_error = self.compress_combin.compress(grads, lambda : False)
            # self.combine_error.update(combin_error)
            return {"combin_alpha": combin_alpha}
            
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
        model_params = self.model.state_dict()
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            regular_model_params={
                key: model_params[key].clone().cpu() for key in self.regular_params_name
            },
            personal_model_params={
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
        )
        if self.return_diff:
            client_package["model_params_diff"] = self.pack_client_model(
                client_package["regular_model_params"],
                self.global_regular_model_params,
            )
            client_package.pop("regular_model_params")
        return client_package