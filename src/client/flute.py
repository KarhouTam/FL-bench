import numpy as np
import torch

from src.client.fedper import FedPerClient
from src.utils.constants import NUM_CLASSES


class FLUTEClient(FedPerClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.num_classes = NUM_CLASSES[self.args.dataset.name]
        self.clients_data_labels = []
        for indices in self.data_indices:
            data_labels = torch.zeros((self.num_classes, 1), device=self.device)
            for y in np.unique(self.dataset.targets[indices["train"]]):
                data_labels[y][0] = 1
            self.clients_data_labels.append(data_labels)

    def package(self):
        client_package = super().package()
        client_package["data_labels"] = self.clients_data_labels[self.client_id].cpu()
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        for E in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                features = self.model.get_last_features(x)
                logit = self.model.classifier(features)
                loss = self.criterion(logit, y)
                weight = self.model.classifier.weight
                loss += self.args.flute.lamda1 * torch.norm(
                    torch.matmul(weight, weight.t())
                    / torch.norm(torch.matmul(weight, weight.t()))
                    - 1
                    / torch.sqrt(torch.tensor(self.num_classes - 1).to(self.device))
                    * torch.mul(
                        (
                            torch.eye(self.num_classes).to(self.device)
                            - 1
                            / self.num_classes
                            * torch.ones((self.num_classes, self.num_classes)).to(
                                self.device
                            )
                        ),
                        torch.matmul(
                            self.clients_data_labels[self.client_id],
                            self.clients_data_labels[self.client_id].t(),
                        ).to(self.device),
                    )
                )
                loss += self.args.flute.lamda2 * torch.norm(weight) ** 2
                if E >= self.local_epoch - self.args.flute.rep_round:
                    loss += self.args.flute.lamda3 * torch.norm(features) ** 2
                self.optimizer.zero_grad()
                loss.backward()
                if E < self.local_epoch - self.args.flute.rep_round:
                    self.model.base.zero_grad()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
