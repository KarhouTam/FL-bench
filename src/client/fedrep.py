import torch

from fedper import FedPerClient
from src.utils.models import DecoupledModel
from src.utils.tools import Logger, NestedNamespace


class FedRepClient(FedPerClient):
    def __init__(
        self,
        model: DecoupledModel,
        args: NestedNamespace,
        logger: Logger,
        device: torch.device,
    ):
        super().__init__(model, args, logger, device)

    def fit(self):
        self.model.train()
        self.dataset.train()
        for E in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                # freeze body, train head
                if E < self.local_epoch - self.args.fedrep.train_body_epoch:
                    for name, param in self.model.named_parameters():
                        if name not in self.personal_params_name:
                            param.grad.zero_()
                # freeze head, train body
                else:
                    for name, param in self.model.named_parameters():
                        if name in self.personal_params_name:
                            param.grad.zero_()
                self.optimizer.step()

    def finetune(self):
        self.model.train()
        self.dataset.train()
        full_model = False
        if full_model:
            # fine-tune the full model
            for E in range(self.args.common.finetune_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue

                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # freeze body, train head
                    if (
                        E
                        < self.args.common.finetune_epoch
                        - self.args.fedrep.train_body_epoch
                    ):
                        for name, param in self.model.named_parameters():
                            if name not in self.personal_params_name:
                                param.grad.zero_()
                    # freeze head, train body
                    else:
                        for name, param in self.model.named_parameters():
                            if name in self.personal_params_name:
                                param.grad.zero_()
        else:
            # fine-tune the classifier only
            for _ in range(self.args.common.finetune_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue

                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for name, param in self.model.named_parameters():
                        if name not in self.personal_params_name:
                            param.grad.zero_()
                    self.optimizer.step()
