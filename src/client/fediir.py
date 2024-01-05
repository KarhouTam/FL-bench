import torch
import torch.nn.functional as F
import torch.autograd as autograd

from fedavg import FedAvgClient


class FedIIRClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super(FedIIRClient, self).__init__(model, args, logger, device)

    def fit(self):
        self.model.train()
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
                loss = loss_erm + self.args.penalty * penalty_value
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def grad(self, client_id, new_parameters):
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)
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
        return grad_sum, total_batch
