from copy import deepcopy
from typing import Any, OrderedDict

import torch
from src.client.fedavg import FedAvgClient
from src.utils.compressor_utils import CompressorCombin
from src.utils.metrics import Metrics


class FedTest5Client(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.compress_combin:CompressorCombin = None
        self.global_model = None
        self.personal_params_name = self.model.state_dict().keys()

    @torch.no_grad()
    def set_parameters(self, package: dict[str, Any]):
        self.compress_combin = package['compress_combin']
        # self.global_model = package['global_model']

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

        last_global_model_params:dict = package["last_global_model_params"]
        global_grad = package["global_grad"]
        if global_grad['combin_alpha'] != {}:
            print(f"Client {self.client_id} Compressing the global_grad...")
            self.compress_combin.update(global_grad['combin_update_dict'])
            global_grad = self.compress_combin.uncompress(global_grad['combin_alpha'], self.model.state_dict())
            self.model.load_state_dict(package["personal_model_params"], strict=False)
            self.model.load_state_dict(last_global_model_params, strict=False)

            model_params = self.model.state_dict()
            for key, value in global_grad.items():
                model_params[key].data -= value.to(model_params[key].device)    
        else:
            print(f"Client {self.client_id} No need to compress the global_grad...")
            self.model.load_state_dict(package["personal_model_params"], strict=False)
            self.model.load_state_dict(package["regular_model_params"], strict=False)
            model_params = self.model.state_dict()
        
        self.global_model = deepcopy(self.model)
        self.global_model.load_state_dict(package["personal_model_params"], strict=False)
        self.global_model.load_state_dict(package["regular_model_params"], strict=False)
        
        self.global_regular_model_params = OrderedDict(
            (key, model_params[key].clone().cpu())
            for key in self.regular_params_name
        )

        last_global_model_params.update(self.global_regular_model_params)


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
        eval_results["before"] = self.evaluate(self.global_model)
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
