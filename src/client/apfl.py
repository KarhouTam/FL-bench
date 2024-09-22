from copy import deepcopy
from typing import Any

import torch

from src.client.fedavg import FedAvgClient


class APFLClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.alpha = torch.tensor(self.args.apfl.alpha, device=self.device)
        self.local_model = deepcopy(self.model)

        def _re_init(src):
            target = deepcopy(src)
            for module in target.modules():
                if (
                    isinstance(module, torch.nn.Conv2d)
                    or isinstance(module, torch.nn.BatchNorm2d)
                    or isinstance(module, torch.nn.Linear)
                ):
                    module.reset_parameters()
            return deepcopy(target.state_dict())

        self.optimizer.add_param_group({"params": self.local_model.parameters()})
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.local_model.load_state_dict(package["local_model_params"])
        self.alpha = package["alpha"].to(self.device)

    def package(self):
        client_parckage = super().package()
        client_parckage["local_model_params"] = {
            key: param.cpu().clone()
            for key, param in self.local_model.state_dict().items()
        }
        client_parckage["alpha"] = self.alpha.cpu().clone()
        return client_parckage

    def fit(self):
        self.model.train()
        self.local_model.train()
        self.dataset.train()
        for i in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit_g = self.model(x)
                loss = self.criterion(logit_g, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                logit_l = self.local_model(x)
                logit_g = self.model(x)
                logit_p = self.alpha * logit_l + (1 - self.alpha) * logit_g.detach()
                loss = self.criterion(logit_p, y)
                loss.backward()
                self.optimizer.step()

                if self.args.apfl.adaptive_alpha and i == 0:
                    self.update_alpha()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    # refers to https://github.com/MLOPTPSU/FedTorch/blob/b58da7408d783fd426872b63fbe0c0352c7fa8e4/fedtorch/comms/utils/flow_utils.py#L240
    def update_alpha(self):
        alpha_grad = 0
        for local_param, global_param in zip(
            self.local_model.parameters(), self.model.parameters()
        ):
            if local_param.requires_grad:
                diff = (local_param.data - global_param.data).flatten()
                grad = (
                    self.alpha * local_param.grad.data
                    + (1 - self.alpha) * global_param.grad.data
                ).flatten()
                alpha_grad += diff @ grad

        alpha_grad += 0.02 * self.alpha
        self.alpha.data -= self.args.optimizer.lr * alpha_grad
        self.alpha.clip_(0, 1.0)

    def evaluate(self):
        return super().evaluate(
            model=MixedModel(self.local_model, self.model, alpha=self.alpha)
        )


class MixedModel(torch.nn.Module):
    def __init__(
        self, local_model: torch.nn.Module, global_model: torch.nn.Module, alpha: float
    ):
        super().__init__()
        self.local_model = local_model
        self.global_model = global_model
        self.alpha = alpha

    def forward(self, x):
        return (
            self.alpha * self.local_model(x)
            + (1 - self.alpha) * self.global_model(x).detach()
        )
