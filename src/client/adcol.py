from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.client.fedavg import FedAvgClient


class ADCOLClient(FedAvgClient):
    def __init__(self, discriminator: torch.nn.Module, client_num: int, **commons):
        super(ADCOLClient, self).__init__(**commons)
        self.discriminator = discriminator.to(self.device)
        self.client_num = client_num
        self.features_list = []

    def fit(self):
        self.model.train()
        self.discriminator.eval()
        self.dataset.train()
        self.features_list = []
        for i in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                try:
                    features = self.model.base(x)
                    logit = self.model.classifier(F.relu(features))
                except:
                    raise ValueError(
                        "model may have no feature extractor + classifier architecture"
                    )
                cross_entropy = self.criterion(logit, y).mean()
                client_index = self.discriminator(features)
                client_index_softmax = F.log_softmax(client_index, dim=-1)
                target_index = torch.full(client_index.shape, 1 / self.client_num).to(
                    self.device
                )
                target_index_softmax = F.softmax(target_index, dim=-1)
                kl_loss_func = nn.KLDivLoss(reduction="batchmean").to(self.device)
                kl_loss = kl_loss_func(client_index_softmax, target_index_softmax)
                mu = self.args.adcol.mu

                loss = cross_entropy + mu * kl_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # collect features in the last epoch
                if i == self.local_epoch - 1:
                    self.features_list.append(features.detach().clone().cpu())

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.feature_list = torch.cat(self.features_list, dim=0)

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.discriminator.load_state_dict(package["new_discriminator_params"])

    def package(self):
        client_package = super().package()
        client_package["features_list"] = self.features_list
        return client_package
