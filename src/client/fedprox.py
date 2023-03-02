from fedavg import FedAvgClient
from src.config.utils import trainable_params


class FedProxClient(FedAvgClient):
    def __init__(self, model, args, logger):
        super(FedProxClient, self).__init__(model, args, logger)

    def train(self, client_id, new_parameters, verbose=False):
        delta, _, stats = super().train(
            client_id, new_parameters, return_diff=True, verbose=verbose
        )
        # FedProx's model aggregation doesn't need weight
        return delta, 1.0, stats

    def fit(self):
        self.model.train()
        global_params = [p.clone().detach() for p in trainable_params(self.model)]
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
                    w.grad.data += self.args.mu * (w.data - w_t.data)
                self.optimizer.step()
