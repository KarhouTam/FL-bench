from src.client.fedavg import FedAvgClient


class FedPerClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.personal_params_name.extend(
            [name for name in self.model.state_dict().keys() if "classifier" in name]
        )

    def finetune(self):
        self.model.train()
        full_model = True
        if full_model:
            # fine-tune the full model
            super().finetune()
        else:
            # fine-tune the classifier only
            for _ in range(self.args.common.test.client.finetune_epoch):
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
