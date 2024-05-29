from src.client.fedper import FedPerClient


class FedRepClient(FedPerClient):
    def __init__(self, **commons):
        super().__init__(**commons)

    def fit(self):
        self.model.train()
        self.dataset.train()
        for E in range(self.local_epoch):
            if E < self.local_epoch - self.args.fedrep.train_body_epoch:
                self.model.classifier.requires_grad_(True)
                self.model.base.requires_grad_(False)
            else:
                self.model.classifier.requires_grad_(False)
                self.model.base.requires_grad_(True)

            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.model.requires_grad_(True)

    def finetune(self):
        self.model.train()
        self.dataset.train()
        full_model = False
        if full_model:
            # fine-tune the full model
            for E in range(self.args.common.finetune_epoch):
                if (
                    E
                    < self.args.common.finetune_epoch
                    - self.args.fedrep.train_body_epoch
                ):
                    self.model.classifier.requires_grad_(True)
                    self.model.base.requires_grad_(False)
                else:
                    self.model.classifier.requires_grad_(False)
                    self.model.base.requires_grad_(True)
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
                    self.optimizer.step()
        else:
            # fine-tune the classifier only
            self.model.base.requires_grad_(False)
            self.model.classifier.requires_grad_(True)
            for _ in range(self.args.common.finetune_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue

                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        self.model.requires_grad_(True)
