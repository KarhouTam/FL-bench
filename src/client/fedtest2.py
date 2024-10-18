from copy import deepcopy
from typing import Any, OrderedDict

import torch
from src.client.fedavg import FedAvgClient
from src.utils.compressor_utils import CompressorCombin


class FedTest2Client(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.compress_combin:CompressorCombin = None
        self.combine_error:dict = None
        self.total_weight = None

    def fit(self):
        # for layer_name, layer_param in self.model.named_parameters():
        #     if layer_name in self.combine_error:
        #         try:
        #             if self.total_weight is not None:
        #                 layer_param.data -= (self.combine_error[layer_name] * len(self.trainset) / self.total_weight).to(layer_param.device)
        #             else:
        #                 layer_param.data -= self.combine_error[layer_name].to(layer_param.device)
        #         except Exception as e:
        #             print(e, layer_name, layer_param.shape, self.combine_error[layer_name].shape)

        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    @torch.no_grad()
    def set_parameters(self, package: dict[str, Any]):
        self.compress_combin = package['compress_combin']
        self.combine_error = package['combin_error']
        self.total_weight = package['total_weight']

        return super().set_parameters(package)        
        
    def pack_client_model(self, raw_model:dict[str, torch.Tensor] , global_model:dict[str, torch.Tensor]):
        packet_to_send = {}
        grads = {}
        for key in raw_model.keys():
            # quantized_model[key], scale[key] = (global_model[key] - raw_model[key]), 0
            grads[key] = global_model[key] - raw_model[key]
        combin_alpha, combin_update_dict, combin_error = self.compress_combin.compress(grads, lambda : True)
        self.combine_error.update(combin_error)
            
        packet_to_send["combin_alpha"] = combin_alpha
        packet_to_send["combin_update_dict"] = combin_update_dict
          
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