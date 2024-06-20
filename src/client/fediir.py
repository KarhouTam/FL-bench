from typing import Any

import torch
import torch.autograd as autograd
import torch.nn.functional as F

from src.client.fedavg import FedAvgClient


class FedIIRClient(FedAvgClient):
    def __init__(self, **commons):
        super(FedIIRClient, self).__init__(**commons)
        self.grad_mean: tuple[torch.Tensor] = None

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.grad_mean = package["grad_mean"]
        if self.grad_mean is not None:
            self.grad_mean = tuple(grad.to(self.device) for grad in self.grad_mean)

    def fit(self):
        self.model.train()
        self.dataset.train()
        for i in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                try:
                    features = self.model.base(x)
                    logit = self.model.classifier(F.relu(features))
                except:
                    print(
                        "model may have no feature extractor + classifier architecture"
                    )
                loss_erm = F.cross_entropy(logit, y)
                grad_client = autograd.grad(
                    loss_erm, self.model.classifier.parameters(), create_graph=True
                )
                penalty_value = 0
                for g_client, g_mean in zip(grad_client, self.grad_mean):
                    penalty_value += (g_client - g_mean).pow(2).sum()
                loss = loss_erm + self.args.fediir.penalty * penalty_value
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def grad(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        grad_sum = tuple(
            torch.zeros_like(p).to(self.device)
            for p in list(self.model.classifier.parameters())
        )
        total_batch = 0
        for x, y in self.trainloader:
            if len(x) <= 1:
                continue

            x, y = x.to(self.device), y.to(self.device)
            try:
                features = self.model.base(x)
                logits = self.model.classifier(F.relu(features))
            except:
                print("model may have no feature extractor + classifier architecture")
            loss = F.cross_entropy(logits, y)
            grad_batch = autograd.grad(
                loss, self.model.classifier.parameters(), create_graph=False
            )
            grad_sum = tuple(g1 + g2 for g1, g2 in zip(grad_sum, grad_batch))
            total_batch += 1
        return dict(
            grad_sum=tuple(grad.detach().cpu().clone() for grad in grad_sum),
            total_batch=total_batch,
        )
