from copy import deepcopy

import torch

from src.client.fedavg import FedAvgClient
from src.utils.tools import count_labels


class FedLCClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.label_distrib = torch.zeros(len(self.dataset.classes), device=self.device)

        def logit_calibrated_loss(logit, y):
            cal_logit = torch.exp(
                logit
                - (
                    self.args.fedlc.tau
                    * torch.pow(self.label_distrib, -1 / 4)
                    .unsqueeze(0)
                    .expand((logit.shape[0], -1))
                )
            )
            y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
            loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
            return loss.sum() / logit.shape[0]

        self.criterion = logit_calibrated_loss

    def load_data_indices(self):
        super().load_data_indices()
        label_counts = count_labels(self.dataset, self.trainset.indices)
        self.label_distrib.zero_()
        for cls, count in enumerate(label_counts):
            self.label_distrib[cls] = max(1e-8, count)
