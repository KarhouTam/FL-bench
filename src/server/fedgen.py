from argparse import ArgumentParser, Namespace
from copy import deepcopy
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fedavg import FedAvgServer, get_fedavg_argparser
from src.client.fedgen import FedGenClient
from src.config.utils import trainable_params


def get_fedgen_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--ensemble_lr", type=float, default=3e-4)
    parser.add_argument("--gen_batch_size", type=int, default=32)
    parser.add_argument("--generative_alpha", type=float, default=10)
    parser.add_argument("--generative_beta", type=float, default=10)
    parser.add_argument("--ensemble_alpha", type=float, default=1)
    parser.add_argument("--ensemble_beta", type=float, default=0)
    parser.add_argument("--ensemble_eta", type=float, default=0)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--embedding", type=int, default=0)
    parser.add_argument("--coef_decay", type=float, default=0.98)
    parser.add_argument("--coef_decay_epoch", type=int, default=1)
    parser.add_argument("--ensemble_epoch", type=int, default=50)
    parser.add_argument("--train_generator_epoch", type=int, default=5)
    parser.add_argument("--min_samples_per_label", type=int, default=1)
    return parser


class FedGenServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedGen",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        if args is None:
            args = get_fedgen_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedGenClient(deepcopy(self.model), self.args, self.logger, self.device)
        self.generator = Generator(args, self).to(self.device)
        self.generator_optimizer = torch.optim.Adam(
            trainable_params(self.generator), args.ensemble_lr
        )
        self.generator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.generator_optimizer, gamma=0.98
        )
        self.unique_labels = range(len(self.trainer.dataset.classes))
        self.teacher_model = deepcopy(self.model)

    def train_one_round(self):
        client_params_cache = []
        weight_cache = []
        label_counts_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)

            (
                delta,
                weight,
                label_counts,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                current_global_epoch=self.current_epoch,
                new_parameters=client_local_params,
                generator=self.generator,
                regularization=self.current_epoch > 0,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )
            label_counts_cache.append(label_counts)
            client_params_cache.append(delta)
            weight_cache.append(weight)
        self.train_generator(client_params_cache, label_counts_cache)
        self.aggregate(client_params_cache, weight_cache)

    @torch.no_grad()
    def aggregate(
        self, client_params_cache: List[List[torch.Tensor]], weight_cache: List[int]
    ):
        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)
        aggregated_params = [
            torch.sum(weights * torch.stack(layer_params, dim=-1), dim=-1)
            for layer_params in zip(*client_params_cache)
        ]

        for old_param, new_param in zip(
            self.global_params_dict.values(), aggregated_params
        ):
            old_param.data.copy_(new_param)

    def train_generator(
        self,
        client_params_cache: List[List[torch.Tensor]],
        label_counts_cache: List[List[int]],
    ):
        self.generator.train()
        self.teacher_model.eval()
        self.model.eval()
        label_weights, qualified_labels = self.get_label_weights(label_counts_cache)
        for _ in range(self.args.train_generator_epoch):
            y_npy = np.random.choice(qualified_labels, self.args.batch_size)
            y_tensor = torch.tensor(y_npy, dtype=torch.long, device=self.device)

            generator_output, eps = self.generator(y_tensor)

            diversity_loss = self.generator.diversity_loss(eps, generator_output)

            teacher_loss = 0
            teacher_logit = 0

            for i, model_params in enumerate(client_params_cache):
                self.teacher_model.load_state_dict(
                    OrderedDict(zip(self.trainable_params_name, model_params)),
                    strict=False,
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
                F.log_softmax(student_logits, dim=1), F.softmax(teacher_logit, dim=1)
            )
            loss = (
                self.args.ensemble_alpha * teacher_loss
                - self.args.ensemble_beta * student_loss
                + self.args.ensemble_eta * diversity_loss
            )
            self.generator_optimizer.zero_grad()
            loss.backward()
            self.generator_optimizer.step()

        self.generator_lr_scheduler.step()

    def get_label_weights(self, label_counts_cache):
        label_weights = []
        qualified_labels = []
        for i, label_counts in enumerate(zip(*label_counts_cache)):
            count_sum = sum(label_counts)
            label_weights.append(np.array(label_counts) / count_sum)
            if count_sum / len(label_counts_cache) > self.args.min_samples_per_label:
                qualified_labels.append(i)
        label_weights = np.array(label_weights).reshape((len(self.unique_labels)), -1)
        return label_weights, qualified_labels


class Generator(nn.Module):
    def __init__(self, args, server: FedGenServer) -> None:
        super().__init__()
        from src.config.models import MODEL_DICT

        self.device = torch.device(
            "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
        )
        # obtain the latent dim
        dummy_model = MODEL_DICT[args.model](args.dataset)
        x = torch.zeros(1, *server.trainer.dataset[0][0].shape)

        self.use_embedding = args.embedding
        self.latent_dim = dummy_model.base(x).shape[-1]
        self.hidden_dim = args.hidden_dim
        self.noise_dim = args.noise_dim
        self.class_num = len(server.trainer.dataset.classes)

        if args.embedding:
            self.embedding = nn.Embedding(self.class_num, args.noise_dim)
        input_dim = (
            self.noise_dim * 2 if args.embedding else self.noise_dim + self.class_num
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
        eps = torch.randn((batchsize, self.noise_dim), device=self.device)
        if self.use_embedding:
            y = self.embedding(targets)
        else:
            y = torch.zeros((batchsize, self.class_num), device=self.device)
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
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=2)
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=2)
        elif metric == 'cosine':
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
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))


if __name__ == "__main__":
    server = FedGenServer()
    server.run()
