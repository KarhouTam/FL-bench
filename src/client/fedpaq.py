from copy import deepcopy
from typing import OrderedDict
import torch
from torch.nn import BatchNorm2d

from src.client.fedavg import FedAvgClient
from src.utils.my_utils import QSGDQuantizer


class FedPAQClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.quantizer = QSGDQuantizer(256)

    def pack_client_model(self, raw_model:dict[str, torch.Tensor] , global_model):
        quantized_model = {}
        packet_to_send = {}
        scale = {}
        for key in raw_model.keys():
            # quantized_model[key], scale[key] = (global_model[key] - raw_model[key]), 0
            if raw_model[key].dtype == torch.long or ('running_var' in key) or ('running_mean' in key):
                quantized_tensor, scale[key] = (global_model[key] - raw_model[key]), 0
                quantized_model[key] = quantized_tensor.to(torch.long)
            else:
                quantized_tensor, min_val, scale[key] = self.quantizer.quantize(global_model[key] - raw_model[key])
                quantized_model[key] = quantized_tensor.to(torch.int8)

        packet_to_send["tensors"] = quantized_model
        packet_to_send["scales"] = scale
        return packet_to_send

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