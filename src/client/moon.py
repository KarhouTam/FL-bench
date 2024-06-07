from copy import deepcopy
from typing import Any

import torch
from torch.nn.functional import cosine_similarity

from src.client.fedavg import FedAvgClient


class MOONClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.prev_model = deepcopy(self.model)
        self.global_model = deepcopy(self.model)

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.global_model.load_state_dict(self.model.state_dict())
        if package["prev_model_params"]:
            self.prev_model.load_state_dict(package["prev_model_params"])
        else:
            self.prev_model.load_state_dict(self.model.state_dict())

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                z_curr = self.model.get_last_features(x, detach=False)
                z_global = self.global_model.get_last_features(x, detach=True)
                z_prev = self.prev_model.get_last_features(x, detach=True)
                logits = self.model.classifier(z_curr)
                loss_sup = self.criterion(logits, y)
                loss_con = -torch.log(
                    torch.exp(
                        cosine_similarity(z_curr.flatten(1), z_global.flatten(1))
                        / self.args.moon.tau
                    )
                    / (
                        torch.exp(
                            cosine_similarity(z_prev.flatten(1), z_curr.flatten(1))
                            / self.args.moon.tau
                        )
                        + torch.exp(
                            cosine_similarity(z_curr.flatten(1), z_global.flatten(1))
                            / self.args.moon.tau
                        )
                    )
                )

                loss = loss_sup + self.args.moon.mu * torch.mean(loss_con)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
