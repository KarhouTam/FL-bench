import copy
from collections import OrderedDict
from typing import Any

import torch

from src.client.fedavg import FedAvgClient


class FedAHClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.global_model = copy.deepcopy(self.model)
        self.head_weights = [
            torch.ones_like(param.data).to(self.device)
            for param in list(self.global_model.classifier.parameters())
        ]

    def fit(self):
        self.model.train()
        self.dataset.train()

        # Train head
        for _ in range(self.args.fedah.plocal_epochs):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                # fix base
                for param in self.model.base.parameters():
                    if param.requires_grad:
                        param.grad.zero_()
                self.optimizer.step()

        # Train base
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                # fix head
                for param in self.model.classifier.parameters():
                    if param.requires_grad:
                        param.grad.zero_()
                self.optimizer.step()

    def set_parameters(self, package: dict[str, Any]):
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()

        if (
            package["optimizer_state"]
            and not self.args.common.reset_optimizer_on_global_epoch
        ):
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        if self.lr_scheduler is not None:
            if package["lr_scheduler_state"]:
                self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])
            else:
                self.lr_scheduler.load_state_dict(self.init_lr_scheduler_state)

        self.global_model.load_state_dict(package["regular_model_params"], strict=False)
        self.global_model.load_state_dict(
            package["personal_model_params"], strict=False
        )

        self.model.base.load_state_dict(
            self.global_model.base.state_dict(), strict=False
        )

        self.aggregate_head()

        if self.args.common.buffers == "drop":
            self.model.load_state_dict(self.init_buffers, strict=False)

        if self.return_diff:
            model_params = self.model.state_dict()
            self.regular_model_params = OrderedDict(
                (key, model_params[key].clone().cpu())
                for key in self.regular_params_name
            )

    def aggregate_head(self):
        self.dataset.train()

        # aggregate head
        # obtain the references of the head parameters
        params_g = list(self.global_model.classifier.parameters())
        params = list(self.model.classifier.parameters())

        # temp local model only for head weights learning
        model_t = copy.deepcopy(self.model)
        params_th = list(model_t.classifier.parameters())
        params_tb = list(model_t.base.parameters())

        # frozen base to reduce computational cost in Pytorch
        for param in params_tb:
            param.requires_grad = False
        for param_t in params_th:
            param_t.requires_grad = True

        # used to obtain the gradient of model, no need to use optimizer.step(), so lr=0
        optimizer_t = torch.optim.SGD(model_t.parameters(), lr=0)

        for param_t, param, param_g, weight in zip(
            params_th, params, params_g, self.head_weights
        ):
            param_t.data = param + (param_g - param) * weight

        for epoch in range(self.args.fedah.plocal_epochs):
            for i, (x, y) in enumerate(self.trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                optimizer_t.zero_grad()
                output = model_t(x)
                loss_value = self.criterion(output, y)
                loss_value.backward()

                # update head weights in this batch
                for param_t, param, param_g, weight in zip(
                    params_th, params, params_g, self.head_weights
                ):
                    weight.data = torch.clamp(
                        weight
                        - self.args.fedah.eta * (param_t.grad * (param_g - param)),
                        0,
                        1,
                    )

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(
                    params_th, params, params_g, self.head_weights
                ):
                    param_t.data = param + (param_g - param) * weight

        # obtain initialized aggregated head
        for param, param_t in zip(params, params_th):
            param.data = param_t.data.clone()
