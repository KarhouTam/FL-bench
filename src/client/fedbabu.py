import torch

from fedavg import FedAvgClient
from src.utils.models import DecoupledModel
from src.utils.tools import Logger, NestedNamespace

class FedBabuClient(FedAvgClient):
    def __init__(self, model: DecoupledModel, args: NestedNamespace, logger: Logger, device: torch.device):
        super().__init__(model, args, logger, device)

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                # fix head(classifier)
                for param in self.model.classifier.parameters():
                    param.grad.zero_()
                self.optimizer.step()
