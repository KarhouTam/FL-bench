from fedavg import FedAvgClient


class FedPerClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        self.personal_params_name = [
            name for name in self.model.state_dict().keys() if "classifier" in name
        ]
        self.init_personal_params_dict = {
            name: param.clone().detach()
            for name, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (name in self.personal_params_name)
        }

    def finetune(self):
        self.model.train()
        full_model = True
        if full_model:
            # fine-tune the full model
            super().finetune()
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
