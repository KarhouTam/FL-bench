from copy import deepcopy
from typing import Dict, OrderedDict

import torch

from fedavg import FedAvgClient
from src.config.utils import trainable_params


class DittoClient(FedAvgClient):
    def __init__(self, model, args, logger, device, client_num):
        super().__init__(model, args, logger, device)
        self.pers_model = deepcopy(model)
        self.pers_model_params_dict = {
            cid: deepcopy(self.pers_model.state_dict()) for cid in range(client_num)
        }
        self.optimizer.add_param_group(
            {"params": trainable_params(self.pers_model), "lr": self.args.local_lr}
        )
        self.init_opt_state_dict = deepcopy(self.optimizer.state_dict())

    def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):
        super().set_parameters(new_parameters)
        self.global_params = new_parameters
        self.pers_model.load_state_dict(self.pers_model_params_dict[self.client_id])

    def save_state(self):
        super().save_state()
        self.pers_model_params_dict[self.client_id] = deepcopy(
            self.pers_model.state_dict()
        )

    def fit(self):
        self.model.train()
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

        for _ in range(self.args.pers_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                logit = self.pers_model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                for pers_param, global_param in zip(
                    trainable_params(self.pers_model),
                    trainable_params(self.global_params),
                ):
                    pers_param.grad.data += self.args.lamda * (
                        pers_param.data - global_param.data
                    )
                self.optimizer.step()

    def evaluate(self) -> Dict[str, float]:
        return super().evaluate(self.pers_model)
