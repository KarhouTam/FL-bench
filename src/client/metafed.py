import random
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.client.fedavg import FedAvgClient
from src.utils.tools import evalutate_model


class MetaFedClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.special_valset = Subset(self.dataset, indices=[])
        self.special_valloader = DataLoader(
            self.special_valset, batch_size=self.args.common.batch_size
        )
        self.personal_params_name = list(self.model.state_dict().keys())
        self.teacher = deepcopy(self.model)
        self.lamda = self.args.metafed.lamda

    def load_data_indices(self):
        train_data_indices = self.data_indices[self.client_id]["train"]
        num_special_valset_samples = int(
            len(train_data_indices) * self.args.metafed.valset_ratio
        )
        random.shuffle(train_data_indices)
        self.special_valset.indices = train_data_indices[:num_special_valset_samples]
        self.trainset.indices = train_data_indices[num_special_valset_samples:]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def warmup(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        self.fit()
        metrics = evalutate_model(
            self.model, self.special_valloader, device=self.device
        )
        return dict(
            client_model_params=OrderedDict(
                (key, param.detach().cpu().clone())
                for key, param in self.model.state_dict().items()
            ),
            client_flag=metrics.accuracy > self.args.metafed.threshold_1,
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
        )

    def set_parameters(self, package: dict[str, Any]):
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()

        if package["optimizer_state"]:
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        if self.lr_scheduler is not None:
            if package["lr_scheduler_state"]:
                self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])
            else:
                self.lr_scheduler.load_state_dict(self.init_lr_scheduler_state)

        self.client_flag = package["client_flag"]
        self.model.load_state_dict(package["student_model_params"])
        if not self.client_flag:
            self.model.load_state_dict(package["teacher_model_params"], strict=False)

    def package(self):
        client_package = super().package()
        client_package.pop("regular_model_params")
        return client_package

    def fit(self):
        self.model.train()
        self.teacher.eval()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)

                stu_feature = self.model.get_last_features(x, detach=False)
                logit = self.model.classifier(F.relu(stu_feature))
                loss = self.criterion(logit, y)
                if self.client_flag:
                    tea_feature = self.teacher.get_last_features(x)
                    loss += self.lamda * F.mse_loss(stu_feature, tea_feature)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def train(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        self.train_with_eval()
        client_package = self.package()
        metrics = evalutate_model(
            self.model, self.special_valloader, device=self.device
        )
        client_package["client_flag"] = metrics.accuracy > self.args.metafed.threshold_1
        return client_package

    def personalize(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        # loading buffer parameters of the student model
        self.teacher.load_state_dict(server_package["student_model_params"])
        self.teacher.load_state_dict(
            server_package["teacher_model_params"], strict=False
        )
        student_metrics = evalutate_model(
            self.model, self.special_valloader, device=self.device
        )
        teacher_metrics = evalutate_model(
            self.teacher, self.special_valloader, device=self.device
        )
        teacher_acc = teacher_metrics.accuracy
        student_acc = student_metrics.accuracy
        if teacher_acc <= student_acc and teacher_acc < self.args.metafed.threshold_2:
            self.lamda = 0
        else:
            self.lamda = (
                (10 ** (min(1, (teacher_acc - student_acc) * 5)))
                / 10
                * self.args.metafed.lamda
            )
        self.train_with_eval()
        return dict(
            client_model_params=OrderedDict(
                (key, param.detach().cpu().clone())
                for key, param in self.model.state_dict().items()
            ),
            eval_results=self.eval_results,
        )
