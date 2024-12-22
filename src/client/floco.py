from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch

from src.client.fedavg import FedAvgClient


class FlocoClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.pers_model = deepcopy(self.model).to(self.device)
        self.optimizer.add_param_group({"params": self.pers_model.parameters()})
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.model.subregion_parameters = package["subregion_parameters"]
        if self.args.floco.pers_epoch > 0:  # Floco+
            self.global_params = OrderedDict(
                (key, param.to(self.device))
                for key, param in package["regular_model_params"].items()
            )
            self.pers_model.load_state_dict(package["personalized_model_params"])

    def package(self):
        client_package = super().package()
        if self.args.floco.pers_epoch > 0:  # Floco+
            client_package["personalized_model_params"] = OrderedDict(
                (key, param.detach().cpu().clone())
                for key, param in self.pers_model.state_dict().items()
            )
        return client_package

    def fit(self):
        common_params = dict(
            dataset=self.dataset,
            dataloader=self.trainloader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            lr_scheduler=self.lr_scheduler,
            device=self.device,
        )
        # Train global solution simplex
        training_loop(model=self.model, local_epoch=self.local_epoch, **common_params)
        if self.args.floco.pers_epoch > 0:  # Floco+
            # Train personalized solution simplex
            training_loop(
                model=self.pers_model,
                local_epoch=self.args.floco.pers_epoch,
                reg_model_params=self.global_params,
                lamda=self.args.floco.lamda,
                **common_params,
            )

    @torch.no_grad()
    def evaluate(self):
        if self.args.floco.pers_epoch > 0:  # Floco+
            return super().evaluate(self.pers_model)
        else:
            return super().evaluate()


def training_loop(
    model,
    dataset,
    dataloader,
    local_epoch,
    optimizer,
    criterion,
    lr_scheduler,
    device,
    reg_model_params=None,
    lamda=1,
):
    model.train()
    dataset.train()
    for _ in range(local_epoch):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logit = model(x)
            loss = criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            if reg_model_params is not None:  # Floco+
                _regularize_pers_model(model, reg_model_params, lamda)
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()


def _regularize_pers_model(model, reg_model_params, lamda):
    for pers_param, global_param in zip(model.parameters(), reg_model_params.values()):
        if pers_param.requires_grad and pers_param.grad is not None:
            pers_param.grad.data += lamda * pers_param.data - global_param.data
