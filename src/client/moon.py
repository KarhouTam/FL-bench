from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import torch
from torch.nn.functional import cosine_similarity, relu

from fedavg import FedAvgClient


class MOONClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        self.prev_params_dict: Dict[int, OrderedDict[str, torch.Tensor]] = {}
        self.prev_model = deepcopy(self.model)
        self.global_model = deepcopy(self.model)

    def save_state(self):
        super().save_state()
        self.prev_params_dict[self.client_id] = deepcopy(self.model.state_dict())

    def set_parameters(self, new_parameters):
        super().set_parameters(new_parameters)
        self.global_model.load_state_dict(self.model.state_dict())
        if self.client_id in self.prev_params_dict.keys():
            self.prev_model.load_state_dict(self.prev_params_dict[self.client_id])
        else:
            self.prev_model.load_state_dict(self.model.state_dict())

    def fit(self):
        self.model.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                z_curr = self.model.get_final_features(x, detach=False)
                z_global = self.global_model.get_final_features(x, detach=True)
                z_prev = self.prev_model.get_final_features(x, detach=True)
                logit = self.model.classifier(relu(z_curr))
                loss_sup = self.criterion(logit, y)
                loss_con = -torch.log(
                    torch.exp(
                        cosine_similarity(z_curr.flatten(1), z_global.flatten(1))
                        / self.args.tau
                    )
                    / (
                        torch.exp(
                            cosine_similarity(z_prev.flatten(1), z_curr.flatten(1))
                            / self.args.tau
                        )
                        + torch.exp(
                            cosine_similarity(z_curr.flatten(1), z_global.flatten(1))
                            / self.args.tau
                        )
                    )
                )

                loss = loss_sup + self.args.mu * torch.mean(loss_con)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
