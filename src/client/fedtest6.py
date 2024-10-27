from copy import deepcopy
from typing import Any, OrderedDict

import torch
from src.client.fedavg import FedAvgClient
from src.utils.compressor_utils import CompressorCombin
from src.utils.metrics import Metrics


class FedTest6Client(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)

        # 用于broadcast的
        self.broadcast_compress_combin:CompressorCombin = None
        self.test_global_model = None
        self.personal_params_name = self.model.state_dict().keys()

        # 用于upload的
        self.upload_compress_combin:CompressorCombin = None
        self.combine_error:dict = None
        self.total_weight = None

    @torch.no_grad()
    def set_parameters(self, package: dict[str, Any]):
        # print(f"Client {self.client_id} set_parameters...")
        # 用于upload的
        self.upload_compress_combin = package['upload_compress_combin']
        self.combine_error = package['combin_error']
        self.total_weight = package['total_weight']

        ############################## 
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
        ############################## 

        # 用于broadcast的
        self.broadcast_compress_combin = package['broadcast_compress_combin']
        global_weight = package["global_weight"]
        if global_weight['combin_alpha'] != {}:
            print(f"Client {self.client_id} uncompressing the global_weight...")
            self.broadcast_compress_combin.update(global_weight['combin_update_dict'])
            global_weight = self.broadcast_compress_combin.uncompress(global_weight['combin_alpha'], self.model.state_dict())
            self.model.load_state_dict(package["personal_model_params"], strict=False)
            self.model.load_state_dict(global_weight, strict=False)
        else:
            print(f"Client {self.client_id} No need to uncompress the global_weight...")
            self.model.load_state_dict(package["personal_model_params"], strict=False)
            self.model.load_state_dict(package["regular_model_params"], strict=False)
        
        model_params = self.model.state_dict()
        
        self.test_global_model = package["test_model"]
        self.test_global_model.load_state_dict(package["personal_model_params"], strict=False)
        self.test_global_model.load_state_dict(package["regular_model_params"], strict=False)
        
        self.global_regular_model_params = OrderedDict(
            (key, model_params[key].clone())
            for key in self.regular_params_name
        )

    # 用于upload的
    def pack_client_model(self, raw_model:dict[str, torch.Tensor] , global_model:dict[str, torch.Tensor]):
        packet_to_send = {}
        grads = {}
        for key in raw_model.keys():
            # quantized_model[key], scale[key] = (global_model[key] - raw_model[key]), 0
            grads[key] = global_model[key] - raw_model[key]
        combin_alpha, combin_update_dict, combin_error = self.upload_compress_combin.compress(grads, lambda : True)
        self.combine_error.update(combin_error)
            
        packet_to_send["combin_alpha"] = combin_alpha
        packet_to_send["combin_update_dict"] = combin_update_dict
          
        return packet_to_send

    # 用于upload的
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
        assert self.return_diff, "Client should return the diff of global and local model parameters."
        model_params = self.model.state_dict()
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            personal_model_params={
                key: model_params[key].clone()
                for key in self.personal_params_name
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),

            model_params_diff=self.pack_client_model(
                {key: model_params[key].clone() for key in self.regular_params_name},
                self.global_regular_model_params,
            )
        )
        return client_package

    # 用于broadcast的
    def train_with_eval(self):
        """Wraps `fit()` with `evaluate()` and collect model evaluation results

        A model evaluation results dict: {
                `before`: {...}
                `after`: {...}
                `message`: "..."
            }
            `before` means pre-local-training.
            `after` means post-local-training
        """
        eval_results = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }
        eval_results["before"] = self.evaluate(self.test_global_model)
        if self.local_epoch > 0:
            self.fit()
            eval_results["after"] = self.evaluate()

        eval_msg = []
        for split, color, flag, subset in [
            ["train", "yellow", self.args.common.eval_train, self.trainset],
            ["val", "green", self.args.common.eval_val, self.valset],
            ["test", "cyan", self.args.common.eval_test, self.testset],
        ]:
            if len(subset) > 0 and flag:
                eval_msg.append(
                    "client [{}] [{}]({})  loss: {:.4f} -> {:.4f}   accuracy: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        color,
                        split,
                        eval_results["before"][split].loss,
                        eval_results["after"][split].loss,
                        eval_results["before"][split].accuracy,
                        eval_results["after"][split].accuracy,
                    )
                )
        eval_results["message"] = eval_msg
        self.eval_results = eval_results
