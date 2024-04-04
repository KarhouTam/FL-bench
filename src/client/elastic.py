from argparse import Namespace
from collections import OrderedDict
from typing import Dict

import torch
from torch._tensor import Tensor

from fedavg import FedAvgClient
from src.utils.models import DecoupledModel
from src.utils.tools import Logger, trainable_params
from src.utils.models import DecoupledModel


class ElasticClient(FedAvgClient):
    def __init__(
        self,
        model: DecoupledModel,
        args: Namespace,
        logger: Logger,
        device: torch.device,
    ):
        super().__init__(model, args, logger, device)
        self.layer_num = len(trainable_params(self.model))
        self.sensitivity: Dict[int, torch.Tensor] = {}

    def train(
        self,
        client_id: int,
        local_epoch: int,
        new_parameters: OrderedDict[str, Tensor],
        return_diff=True,
        verbose=False,
    ):
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(new_parameters)

        if self.client_id not in self.sensitivity:
            self.sensitivity[self.client_id] = torch.zeros(
                self.layer_num, device=self.device
            )
        self.model.eval()
        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            grads_norm = [
                torch.norm(layer_grad[0]) ** 2
                for layer_grad in torch.autograd.grad(
                    loss, trainable_params(self.model)
                )
            ]
            for i in range(len(grads_norm)):
                self.sensitivity[self.client_id][i] = (
                    self.args.elastic.mu * self.sensitivity[self.client_id][i]
                    + (1 - self.args.elastic.mu) * grads_norm[i].abs()
                )

        eval_results = self.train_and_log(verbose=verbose)

        delta = OrderedDict()
        for (name, p0), p1 in zip(new_parameters.items(), trainable_params(self.model)):
            delta[name] = p0 - p1

        return (
            delta,
            len(self.trainset),
            eval_results,
            self.sensitivity[self.client_id],
        )
