from src.client.fedavg import FedAvgClient


class FedBabuClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)

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
                    if param.requires_grad:
                        param.grad.zero_()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
