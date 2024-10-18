from typing import Any, OrderedDict
from src.client.fedavg import FedAvgClient


class FedTest1Client(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.personal_params_name = self.model.state_dict().keys()
        # print(f"personal_params_name: {self.personal_params_name}, regular_params_name: {self.regular_params_name}")

    def set_parameters(self, package: dict[str, Any]):
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

        if self.return_diff:
            model_params = self.model.state_dict()
            self.global_regular_model_params = OrderedDict(
                (key, model_params[key].clone().cpu())
                for key in self.regular_params_name
            )
