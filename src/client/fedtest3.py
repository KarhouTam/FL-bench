from copy import deepcopy
from typing import Any, OrderedDict

import torch
from src.client.fedavg import FedAvgClient
from src.utils.compressor_utils import CompressorCombin


class FedTest4Client(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.compress_combin:CompressorCombin = None
        self.combine_error:dict = None
        self.personal_params_name = self.model.state_dict().keys()

    @torch.no_grad()
    def set_parameters(self, package: dict[str, Any]):
        self.compress_combin = package['compress_combin']
        self.combine_error = package['combin_error']

        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()

        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        if self.lr_scheduler is not None:
            if package["lr_scheduler_state"]:
                self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])
            else:
                self.lr_scheduler.load_state_dict(self.init_lr_scheduler_state)

        self.model.load_state_dict(package["personal_model_params"], strict=False)
        self.model.load_state_dict(package["regular_model_params"], strict=False)
        
        # 对于新广播的全局梯度, 清空误差
        for key in package["regular_model_params"].keys():
            self.combine_error[key] = torch.zeros_like(self.combine_error[key])

        # 对于广播norm，更新本地模型的长度
        for key, norm in package["norm_broadcast"].items():
            params = self.model.state_dict()[key]
            # 将params的norm更新为广播的norm
            params.data = params.data / params.norm() * norm

        if self.return_diff:
            model_params = self.model.state_dict()
            self.global_regular_model_params = OrderedDict(
                (key, model_params[key].clone().cpu())
                for key in model_params
            )
      
        
    def pack_client_model(self, raw_model:dict[str, torch.Tensor] , global_model:dict[str, torch.Tensor]):
        packet_to_send = {}
        grads = {}
        for key in raw_model.keys():
            # quantized_model[key], scale[key] = (global_model[key] - raw_model[key]), 0
            grads[key] = global_model[key] - raw_model[key]
            if key in self.combine_error:
                grads[key] += self.combine_error[key]
        
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
            # regular_model_params={
            #     key: model_params[key].clone().cpu() for key in self.regular_params_name
            # },
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
                client_package["personal_model_params"],
                self.global_regular_model_params,
            )
            # client_package.pop("regular_model_params")
        return client_package