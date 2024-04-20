from src.client.fedavg import FedAvgClient
from src.utils.tools import trainable_params


class FedProxClient(FedAvgClient):
    def __init__(self, **commons):
        super(FedProxClient, self).__init__(**commons)

    def fit(self):
        self.model.train()
        self.dataset.train()
        global_params = trainable_params(self.model, detach=True)
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                for w, w_t in zip(trainable_params(self.model), global_params):
                    w.grad.data += self.args.fedprox.mu * (w.data - w_t.data)
                self.optimizer.step()
