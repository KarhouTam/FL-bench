from copy import deepcopy
from typing import Dict, OrderedDict
import torch

from fedavg import FedAvgClient
from src.config.utils import trainable_params


class APFLClient(FedAvgClient):
    def __init__(self, model, args, logger, client_num):
        super().__init__(model, args, logger)

        self.alpha_list = [
            torch.tensor(self.args.alpha, device=self.device) for _ in range(client_num)
        ]
        self.alpha = torch.tensor(self.args.alpha, device=self.device)

        self.local_model = deepcopy(self.model)

        def re_init(src):
            target = deepcopy(src)
            for module in target.modules():
                if (
                    isinstance(module, torch.nn.Conv2d)
                    or isinstance(module, torch.nn.BatchNorm2d)
                    or isinstance(module, torch.nn.Linear)
                ):
                    module.reset_parameters()
            return deepcopy(target.state_dict())

        self.local_params_dict: Dict[int, OrderedDict[str, torch.Tensor]] = {
            # cid: re_init(self.model) for cid in range(client_num)
            cid: deepcopy(self.model.state_dict())
            for cid in range(client_num)
        }

        self.optimizer.add_param_group(
            {"params": trainable_params(self.local_model), "lr": self.local_lr}
        )

    def set_parameters(self, new_parameters: OrderedDict[str, torch.nn.Parameter]):
        super().set_parameters(new_parameters)
        self.local_model.load_state_dict(self.local_params_dict[self.client_id])
        self.alpha = self.alpha_list[self.client_id]

    def save_state(self):
        super().save_state()
        self.local_params_dict[self.client_id] = deepcopy(self.local_model.state_dict())
        self.alpha_list[self.client_id] = self.alpha.clone()

    def fit(self):
        self.model.train()
        self.local_model.train()
        for i in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit_g = self.model(x)
                loss = self.criterion(logit_g, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                logit_l = self.local_model(x)
                logit_g = self.model(x)
                logit_p = self.alpha * logit_l + (1 - self.alpha) * logit_g
                loss = self.criterion(logit_p, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.args.adaptive_alpha and i == 0:
                    self.update_alpha()

    # refers to https://github.com/MLOPTPSU/FedTorch/blob/b58da7408d783fd426872b63fbe0c0352c7fa8e4/fedtorch/comms/utils/flow_utils.py#L240
    def update_alpha(self):
        alpha_grad = 0
        for local_param, global_param in zip(
            trainable_params(self.local_model), trainable_params(self.model)
        ):
            diff = (local_param.data - global_param.data).flatten()
            grad = (
                self.alpha * local_param.grad.data
                + (1 - self.alpha) * global_param.grad.data
            ).flatten()
            alpha_grad += diff @ grad

        alpha_grad += 0.02 * self.alpha
        self.alpha.data -= self.local_lr * alpha_grad
        self.alpha.clip_(0, 1.0)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Dict[str, float]]:
        self.model.eval()
        self.local_model.eval()
        train_loss, test_loss = 0, 0
        train_correct, test_correct = 0, 0
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if len(self.testset) > 0 and self.args.eval_test:
            for x, y in self.testloader:
                x, y = x.to(self.device), y.to(self.device)
                logit_g = self.model(x)
                logit_l = self.local_model(x)
                logit_p = self.alpha * logit_l + (1 - self.alpha) * logit_g
                test_loss += criterion(logit_p, y).item()
                pred = torch.argmax(logit_p, -1)
                test_correct += (pred == y).sum().item()

        if len(self.trainset) > 0 and self.args.eval_train:
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                logit_g = self.model(x)
                logit_l = self.local_model(x)
                logit_p = self.alpha * logit_l + (1 - self.alpha) * logit_g
                train_loss += criterion(logit_p, y).item()
                pred = torch.argmax(logit_p, -1)
                train_correct += (pred == y).sum().item()

        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_correct": train_correct,
            "test_correct": test_correct,
            "train_size": float(max(len(self.trainset), 1)),
            "test_size": float(max(len(self.testset), 1)),
        }
