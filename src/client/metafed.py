from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn.functional as F

from fedavg import FedAvgClient
from torch.utils.data import Subset, DataLoader
from src.config.utils import trainable_params, evalutate_model


class MetaFedClient(FedAvgClient):
    def __init__(self, model, args, logger, device, client_num):
        super().__init__(model, args, logger, device)
        self.client_flags = [False for _ in range(client_num)]
        self.valset = Subset(self.dataset, indices=[])
        self.valloader: DataLoader = None
        self.teacher = deepcopy(self.model)
        self.lamda = self.args.lamda

    def load_dataset(self):
        super().load_dataset()
        num_val_samples = int(len(self.trainset) * self.args.valset_ratio)
        self.valset.indices = self.trainset.indices[:num_val_samples]
        self.trainset.indices = self.trainset.indices[num_val_samples:]
        self.valloader = DataLoader(self.valset, 32, shuffle=True)

    def warmup(self, client_id, new_parameters):
        self.client_id = client_id
        self.set_parameters(new_parameters)
        self.load_dataset()
        self.fit()
        self.save_state()
        self.update_flag()
        return trainable_params(self.model, detach=True)

    def update_flag(self):
        _, val_correct, val_sample_num = evalutate_model(
            self.model, self.valloader, device=self.device
        )
        val_acc = val_correct / val_sample_num
        self.client_flags[self.client_id] = val_acc > self.args.threshold_1

    def train(
        self,
        client_id: int,
        local_epoch: int,
        student_parameters: OrderedDict[str, torch.Tensor],
        teacher_parameters: OrderedDict[str, torch.Tensor],
        verbose=False,
    ):
        self.client_id = client_id
        self.local_epoch = local_epoch
        if self.client_flags[self.client_id]:
            self.set_parameters(student_parameters)
        else:
            self.set_parameters(teacher_parameters)
        self.teacher.load_state_dict(teacher_parameters, strict=False)
        self.load_dataset()
        stats = self.train_and_log(verbose)
        return trainable_params(self.model, detach=True), stats

    def fit(self):
        self.teacher.eval()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)

                stu_feature = self.model.get_final_features(x, detach=False)
                logit = self.model.classifier(F.relu(stu_feature))
                loss = self.criterion(logit, y)
                if self.client_flags[self.client_id]:
                    tea_feature = self.teacher.get_final_features(x)
                    loss += self.lamda * F.mse_loss(stu_feature, tea_feature)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def personalize(
        self, client_id, student_parameters, teacher_parameters, verbose=False
    ):
        self.client_id = client_id
        self.set_parameters(student_parameters)
        self.load_dataset()
        self.teacher.load_state_dict(teacher_parameters, strict=False)

        _, student_correct, val_sample_num = evalutate_model(
            self.model, self.valloader, device=self.device
        )
        _, teacher_correct, _ = evalutate_model(
            self.teacher, self.valloader, device=self.device
        )
        teacher_acc = teacher_correct / val_sample_num
        student_acc = student_correct / val_sample_num
        if teacher_acc <= student_acc and teacher_acc < self.args.threshold_2:
            self.lamda = 0
        else:
            self.lamda = (
                (10 ** (min(1, (teacher_acc - student_acc) * 5))) / 10 * self.args.lamda
            )
        stats = self.train_and_log(verbose)
        return trainable_params(self.model, detach=True), stats
