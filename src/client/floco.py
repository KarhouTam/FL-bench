from copy import deepcopy
from collections import OrderedDict
from typing import Any

from src.client.fedavg import FedAvgClient

class FlocoClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.pers_model = deepcopy(self.model).to(self.device)
        self.optimizer.add_param_group({"params": self.pers_model.parameters()})
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.sample_from = package["sample_from"]
        self.subregion_parameters = package["subregion_parameters"]
        self.global_params = OrderedDict(
            (key, param.to(self.device))
            for key, param in package["regular_model_params"].items()
        ).values()
        self.pers_model.load_state_dict(package["personalized_model_params"])

    def package(self):
        client_package = super().package()
        client_package["personalized_model_params"] = OrderedDict(
            (key, param.detach().cpu().clone())
            for key, param in self.pers_model.state_dict().items()
        )
        return client_package

    def fit(self):
        # Train global solution simplex (subregion)
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x, self.sample_from, self.subregion_parameters)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        # Train personalized solution simplex (subregion)
        for _ in range(self.args.floco.pers_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                logit = self.pers_model(x, self.sample_from, self.subregion_parameters)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                for pers_param, global_param in zip(
                    self.pers_model.parameters(), self.global_params
                ):
                    if pers_param.requires_grad:
                        try:
                            pers_param.grad.data += self.args.floco.lamda * (
                                pers_param.data - global_param.data
                                )
                        except:
                            pass
                self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()