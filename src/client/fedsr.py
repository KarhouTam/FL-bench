import torch
import torch.nn.functional as F

from src.client.fedavg import FedAvgClient


class FedSRClient(FedAvgClient):
    def __init__(self, **commons):
        super(FedSRClient, self).__init__(**commons)

    def fit(self):
        self.model.train()
        self.dataset.train()
        for i in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                try:
                    z, (z_mu, z_sigma) = self.model.featurize(x, return_dist=True)
                    logits = self.model.classifier(z)

                except:
                    print(
                        "model may have no feature extractor + classifier architecture"
                    )
                loss = F.cross_entropy(logits, y)
                obj = loss
                regL2R = torch.zeros_like(obj)
                regCMI = torch.zeros_like(obj)
                regNegEnt = torch.zeros_like(obj)
                if self.args.fedsr.L2R_coeff != 0.0:
                    regL2R = z.norm(dim=1).mean()
                    obj = obj + self.args.fedsr.L2R_coeff * regL2R
                if self.args.fedsr.CMI_coeff != 0.0:
                    r_sigma_softplus = F.softplus(self.model.r_sigma)
                    r_mu = self.model.r_mu[y]
                    r_sigma = r_sigma_softplus[y]
                    z_mu_scaled = z_mu * self.model.C
                    z_sigma_scaled = z_sigma * self.model.C
                    regCMI = (
                        torch.log(r_sigma)
                        - torch.log(z_sigma_scaled)
                        + (z_sigma_scaled**2 + (z_mu_scaled - r_mu) ** 2)
                        / (2 * r_sigma**2)
                        - 0.5
                    )
                    regCMI = regCMI.sum(1).mean()
                    obj = obj + self.args.fedsr.CMI_coeff * regCMI
                self.optimizer.zero_grad()
                obj.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
