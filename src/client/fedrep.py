from fedper import FedPerClient


class FedRepClient(FedPerClient):
    def __init__(self, model, args, logger):
        super().__init__(model, args, logger)

    def train(self, client_id, new_parameters, return_diff=True, verbose=False):
        delta, _, stats = super().train(
            client_id, new_parameters, return_diff=return_diff, verbose=verbose
        )
        # FedRep's model aggregation doesn't use weight
        return delta, 1.0, stats

    def fit(self):
        self.model.train()
        for E in range(self.local_epoch + self.args.train_body_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                # freeze body, train head
                if E < self.local_epoch:
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
        full_model = True
        if full_model:
            # fine-tune the full model
            for E in range(self.args.finetune_epoch + self.args.train_body_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue

                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # freeze body, train head
                    if E < self.args.finetune_epoch:
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
            for _ in range(self.args.finetune_epoch):
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
