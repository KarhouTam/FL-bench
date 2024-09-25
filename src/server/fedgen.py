from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.client.fedgen import FedGenClient
from src.server.fedavg import FedAvgServer
from src.utils.constants import DATA_SHAPE, NUM_CLASSES


class FedGenServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--ensemble_lr", type=float, default=3e-4)
        parser.add_argument("--gen_batch_size", type=int, default=32)
        parser.add_argument("--generative_alpha", type=float, default=10)
        parser.add_argument("--generative_beta", type=float, default=10)
        parser.add_argument("--ensemble_alpha", type=float, default=1)
        parser.add_argument("--ensemble_beta", type=float, default=0)
        parser.add_argument("--ensemble_eta", type=float, default=0)
        parser.add_argument("--noise_dim", type=int, default=32)
        parser.add_argument("--hidden_dim", type=int, default=32)
        parser.add_argument("--use_embedding", type=int, default=0)
        parser.add_argument("--coef_decay", type=float, default=0.98)
        parser.add_argument("--coef_decay_epoch", type=int, default=1)
        parser.add_argument("--ensemble_epoch", type=int, default=50)
        parser.add_argument("--train_generator_epoch", type=int, default=5)
        parser.add_argument("--min_samples_per_label", type=int, default=1)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedGen",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.generator = Generator(self)
        self.init_trainer(FedGenClient, generator=self.generator)
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), self.args.fedgen.ensemble_lr
        )
        self.generator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.generator_optimizer, gamma=0.98
        )
        self.unique_labels = range(NUM_CLASSES[self.args.dataset.name])
        self.teacher_model = deepcopy(self.model)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["regularization"] = self.current_epoch > 0
        server_package["generator_params"] = OrderedDict(
            (key, param.detach().cpu().clone())
            for key, param in self.generator.state_dict().items()
        )
        server_package["alpha"] = self.exp_coef_scheduler(
            self.args.fedgen.generative_alpha
        )
        server_package["beta"] = self.exp_coef_scheduler(
            self.args.fedgen.generative_beta
        )
        return server_package

    def train_one_round(self):
        client_packages = self.trainer.train()
        self.train_generator(client_packages)
        self.aggregate(client_packages)

    def train_generator(self, client_packages: dict[int, dict[str, Any]]):
        self.generator.train()
        self.teacher_model.eval()
        self.model.eval()
        self.generator.to(self.device)
        self.model.to(self.device)
        self.teacher_model.to(self.device)
        label_weights, qualified_labels = self.get_label_weights(
            [package["label_counts"] for package in client_packages.values()]
        )
        for _ in range(self.args.fedgen.train_generator_epoch):
            y_npy = np.random.choice(qualified_labels, self.args.common.batch_size)
            y_tensor = torch.tensor(y_npy, dtype=torch.long, device=self.device)

            generator_output, eps = self.generator(y_tensor)

            diversity_loss = self.generator.diversity_loss(eps, generator_output)

            teacher_loss = 0
            teacher_logit = 0

            for i, package in enumerate(client_packages.values()):
                self.teacher_model.load_state_dict(
                    package["regular_model_params"], strict=False
                )
                weight = label_weights[y_npy][:, i].reshape(-1, 1)
                expand_weight = np.tile(weight, (1, len(self.unique_labels)))
                logits = self.model.classifier(generator_output)
                teacher_loss += torch.mean(
                    self.generator.ce_loss(logits, y_tensor)
                    * torch.tensor(weight, dtype=torch.float, device=self.device)
                )
                teacher_logit += logits * torch.tensor(
                    expand_weight, dtype=torch.float, device=self.device
                )

            student_logits = self.model.classifier(generator_output)
            student_loss = F.kl_div(
                F.log_softmax(student_logits, dim=1),
                F.softmax(teacher_logit, dim=1),
                reduction="batchmean",
            )
            loss = (
                self.args.fedgen.ensemble_alpha * teacher_loss
                - self.args.fedgen.ensemble_beta * student_loss
                + self.args.fedgen.ensemble_eta * diversity_loss
            )
            self.generator_optimizer.zero_grad()
            loss.backward()
            self.generator_optimizer.step()

        self.generator_lr_scheduler.step()
        self.generator.cpu()
        self.model.cpu()
        self.teacher_model.cpu()

    def get_label_weights(self, clients_label_counts):
        label_weights = []
        qualified_labels = []
        for i, label_counts in enumerate(zip(*clients_label_counts)):
            count_sum = max(sum(label_counts), 1e-8)
            label_weights.append(np.array(label_counts) / count_sum)
            if (
                count_sum / len(clients_label_counts)
                > self.args.fedgen.min_samples_per_label
            ):
                qualified_labels.append(i)
        label_weights = np.array(label_weights).reshape((len(self.unique_labels)), -1)
        return label_weights, qualified_labels

    def exp_coef_scheduler(self, init_coef):
        return max(
            1e-4,
            init_coef
            * (
                self.args.fedgen.coef_decay
                ** (self.current_epoch // self.args.fedgen.coef_decay_epoch)
            ),
        )


class Generator(nn.Module):
    def __init__(self, server: FedGenServer) -> None:
        super().__init__()
        # obtain the latent dim
        x = torch.zeros(1, *DATA_SHAPE[server.args.dataset.name])
        self.use_embedding = server.args.fedgen.use_embedding
        self.latent_dim = server.model.base(x).shape[-1]
        self.hidden_dim = server.args.fedgen.hidden_dim
        self.noise_dim = server.args.fedgen.noise_dim
        self.class_num = NUM_CLASSES[server.args.dataset.name]

        if server.args.fedgen.use_embedding:
            self.embedding = nn.Embedding(self.class_num, server.args.fedgen.noise_dim)
        input_dim = (
            self.noise_dim * 2
            if server.args.fedgen.use_embedding
            else self.noise_dim + self.class_num
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.latent_layer = nn.Linear(self.hidden_dim, self.latent_dim)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.diversity_loss = DiversityLoss(metric="l1")
        self.dist_loss = nn.MSELoss()

    def forward(self, targets):
        batchsize = targets.shape[0]
        eps = torch.randn((batchsize, self.noise_dim), device=targets.device)
        if self.use_embedding:
            y = self.embedding(targets)
        else:
            y = torch.zeros((batchsize, self.class_num), device=targets.device)
            y.scatter_(1, targets.reshape(-1, 1), 1)
        z = torch.cat([eps, y], dim=1)
        z = self.mlp(z)
        z = self.latent_layer(z)
        return z, eps


class DiversityLoss(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        if metric == "l1":
            return torch.abs(tensor1 - tensor2).mean(dim=2)
        elif metric == "l2":
            return torch.pow(tensor1 - tensor2, 2).mean(dim=2)
        elif metric == "cosine":
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how="l2")
        return torch.exp(torch.mean(-noise_dist * layer_dist))
