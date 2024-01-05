from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
import torch.nn as nn

from fedavg import FedAvgClient
from src.utils.tools import trainable_params


class ADCOLClient(FedAvgClient):
    def __init__(self, model, discriminator, args, logger, device, client_num):
        super(ADCOLClient, self).__init__(model, args, logger, device)
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.client_num = client_num

    def fit(self):
        self.model.train()
        self.discriminator.eval()
        self.featrure_list = []
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
                mu = self.args.mu

                loss = cross_entropy + mu * kl_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # collect features in the last epoch
                if i == self.local_epoch - 1:
                    self.featrure_list.append(features.detach().clone().cpu())

        self.feature_list = torch.cat(self.featrure_list, dim=0)

    def train(
        self,
        client_id: int,
        local_epoch: int,
        new_parameters: OrderedDict[str, torch.Tensor],
        new_discriminator_parameters: Dict[str, torch.Tensor],
        return_diff=True,
        verbose=False,
    ) -> Tuple[
        Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict, List
    ]:
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(new_parameters)
        self.discriminator.load_state_dict(new_discriminator_parameters)
        eval_stats = self.train_and_log(verbose=verbose)

        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1

            return delta, len(self.trainset), eval_stats, self.featrure_list
        else:
            return (
                trainable_params(self.model, detach=True),
                len(self.trainset),
                eval_stats,
                self.featrure_list,
            )
