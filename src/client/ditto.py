from collections import OrderedDict
from copy import deepcopy
from typing import Any

from src.client.fedavg import FedAvgClient


class DittoClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.pers_model = deepcopy(self.model).to(self.device)
        self.optimizer.add_param_group({"params": self.pers_model.parameters()})
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.global_params = OrderedDict(
            (key, param.to(self.device))
            for key, param in package["regular_model_params"].items()
        )
        self.pers_model.load_state_dict(package["personalized_model_params"])

    def package(self):
        client_package = super().package()
        client_package["personalized_model_params"] = OrderedDict(
            (key, param.detach().cpu().clone())
            for key, param in self.pers_model.state_dict().items()
        )
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        for _ in range(self.args.ditto.pers_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                logit = self.pers_model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                for pers_param, global_param in zip(
                    self.pers_model.parameters(), self.global_params.values()
                ):
                    if pers_param.requires_grad:
                        pers_param.grad.data += self.args.ditto.lamda * (
                            pers_param.data - global_param.data
                        )
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def evaluate(self):
        return super().evaluate(self.pers_model)
