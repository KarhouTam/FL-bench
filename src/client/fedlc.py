from collections import Counter

import numpy as np
import torch

from src.client.fedavg import FedAvgClient


class FedLCClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.clients_label_counts = []
        for indices in self.data_indices:
            counter = Counter(np.array(self.dataset.targets)[indices["train"]])
            self.clients_label_counts.append(
                torch.tensor(
                    [counter.get(i, 1e-8) for i in range(len(self.dataset.classes))],
                    device=self.device,
                )
            )

        def logit_calibrated_loss(logit, y):
            cal_logit = torch.exp(
                logit
                - (
                    self.args.fedlc.tau
                    * torch.pow(self.clients_label_counts[self.client_id], -1 / 4)
                    .unsqueeze(0)
                    .expand((logit.shape[0], -1))
                )
            )
            y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
            loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
            return loss.sum() / logit.shape[0]

        self.criterion = logit_calibrated_loss
