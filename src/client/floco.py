from copy import deepcopy
from collections import OrderedDict
from typing import Any
import torch
from src.utils.metrics import Metrics
from src.client.fedavg import FedAvgClient

class FlocoClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.pers_model = deepcopy(self.model).to(self.device)
        self.optimizer.add_param_group({"params": self.pers_model.parameters()})
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())
        self.floco_plus = True if self.args.floco.pers_epoch > 0 else False

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        if package["subregion_parameters"]:
            self.model.set_subregion(package["sample_from"], package["subregion_parameters"])
        self.global_params = OrderedDict(
            (key, param.to(self.device))
            for key, param in package["regular_model_params"].items()
        ).values()
        if self.floco_plus:
            self.pers_model.load_state_dict(package["personalized_model_params"])

    def package(self):
        client_package = super().package()
        if self.floco_plus:
            client_package["personalized_model_params"] = OrderedDict(
                (key, param.detach().cpu().clone())
                for key, param in self.pers_model.state_dict().items()
            )
        return client_package

    def fit(self):
        # Train global solution simplex (subregion)
        self.model.train()
        self.dataset.train()
        training_loop(self.model, self.dataset, self.trainloader, self.local_epoch, 
                      self.optimizer, self.criterion, self.lr_scheduler, self.device)
        if self.floco_plus:
            # Train personalized solution simplex (subregion)
            training_loop(self.pers_model, self.dataset, self.trainloader, self.args.floco.pers_epoch, self.optimizer, 
                          self.criterion, self.lr_scheduler, self.device, self.global_params, self.args.floco.lamda)
                

def training_loop(model, dataset, dataloader, local_epoch, optimizer, 
                  criterion, lr_scheduler, device, reg_model_params=None, lamda=1):
    model.train()
    dataset.train()
    for _ in range(local_epoch):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logit = model(x)
            optimizer.zero_grad()
            loss = criterion(logit, y)
            if reg_model_params is not None:
                for pers_param, global_param in zip(
                    model.parameters(), reg_model_params
                ):
                    if pers_param.requires_grad:
                        try:
                            pers_param.grad.data += lamda * (
                                pers_param.data - global_param.data
                                )
                        except:
                            pass
            loss.backward()
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
    