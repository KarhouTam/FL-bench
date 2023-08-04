from collections import Counter

import torch

from fedavg import FedAvgClient


class FedLCClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        self.label_distrib = torch.zeros(len(self.dataset.classes), device=self.device)

        def logit_calibrated_loss(logit, y):
            cal_logit = torch.exp(
                logit
                - (
                    self.args.tau
                    * torch.pow(self.label_distrib, -1 / 4)
                    .unsqueeze(0)
                    .expand((logit.shape[0], -1))
                )
            )
            y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
            loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
            return loss.sum() / logit.shape[0]

        self.criterion = logit_calibrated_loss

    def load_dataset(self):
        super().load_dataset()
        label_counter = Counter(self.dataset.targets[self.trainset.indices].tolist())
        self.label_distrib.zero_()
        for cls, count in label_counter.items():
            self.label_distrib[cls] = max(1e-8, count)
