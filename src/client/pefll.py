from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src.client.fedavg import FedAvgClient
from src.utils.constants import INPUT_CHANNELS, NUM_CLASSES


class PeFLLClient(FedAvgClient):
    def __init__(self, embed_net, hyper_net, **commons):
        super().__init__(**commons)
        self.embed_net: EmbedNetwork = deepcopy(embed_net).to(self.device)
        self.hyper_net: HyperNetwork = deepcopy(hyper_net).to(self.device)

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)

        # calculate client personalized embedding and generate model parameters
        self.embed_net.load_state_dict(package["embed_net_params"])
        self.hyper_net.load_state_dict(package["hyper_net_params"])
        embedding = torch.zeros(self.args.pefll.embed_dim, device=self.device)
        size = 0
        for i, (x, y) in enumerate(self.trainloader):
            embedding += self.embed_net(x.to(self.device), y.to(self.device)).sum(dim=0)
            size += len(x)
            if i + 1 == self.args.pefll.embed_num_batches:
                break

        embedding /= size

        embedding = (embedding - embedding.mean()) / embedding.std()
        self.regular_model_params = self.hyper_net(embedding)

        # if common.buffers is local, means self.model loads personal buffers already (in super().set_parameters())
        # and hyper net doesn't responsible for the buffers
        self.model.load_state_dict(self.regular_model_params, strict=False)

    # The only difference between this fit() and FedAvgClient's is PFeLL applies grad norm cliping
    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.pefll.clip_norm
                )
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def package(self):
        client_package = super().package()
        # calculate embed net and hyper net gradients
        # in pesudo codes, gradient calculation should be done at the server side
        # but this change won't break the algorithm workflow
        num_embed_net_params = len(list(self.embed_net.parameters()))
        joint_grads = torch.autograd.grad(
            list(self.regular_model_params.values()),
            list(self.embed_net.parameters()) + list(self.hyper_net.parameters()),
            grad_outputs=[
                (param_old - param_new).detach()
                for param_new, param_old in zip(
                    self.model.parameters(), self.regular_model_params.values()
                )
            ],
            allow_unused=True,
        )
        client_package["embed_net_grads"] = [
            grad.cpu() for grad in joint_grads[:num_embed_net_params]
        ]
        client_package["hyper_net_grads"] = [
            grad.cpu() for grad in joint_grads[num_embed_net_params:]
        ]
        return client_package


class EmbedNetwork(nn.Module):
    def __init__(self, args):
        super(EmbedNetwork, self).__init__()
        self.args = args

        in_channels = (
            INPUT_CHANNELS[self.args.dataset.name]
            + bool(self.args.pefll.embed_y) * NUM_CLASSES[self.args.dataset.name]
        )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, self.args.pefll.embed_num_kernels, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                self.args.pefll.embed_num_kernels,
                2 * self.args.pefll.embed_num_kernels,
                5,
            ),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * self.args.pefll.embed_num_kernels * 5 * 5, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, self.args.pefll.embed_dim),
        )
        self.resize = transforms.Resize((32, 32))

    def forward(self, x, y):
        if self.args.pefll.embed_y:
            h, w = x.shape[2], x.shape[3]
            if h < 32 or w < 32:
                x = self.resize(x)
            y = F.one_hot(y, NUM_CLASSES[self.args.dataset.name])
            y = y.view(y.shape[0], y.shape[1], 1, 1)
            c = torch.zeros(
                (x.shape[0], y.shape[1], x.shape[2], x.shape[3]), device=x.device
            )
            c += y
            x = torch.cat((x, c), dim=1)
        return self.model(x)


class HyperNetwork(nn.Module):
    def __init__(self, backbone: nn.Module, args: Namespace):
        super().__init__()
        self.args = args

        mlp_layers = [
            nn.Linear(self.args.pefll.embed_dim, self.args.pefll.hyper_hidden_dim)
        ]
        for _ in range(self.args.pefll.hyper_num_hidden_layers):
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(
                nn.Linear(
                    self.args.pefll.hyper_hidden_dim, self.args.pefll.hyper_hidden_dim
                )
            )
        self.mlp = nn.Sequential(*mlp_layers)

        parameters, self.params_name = [], []
        for key, param in backbone.named_parameters():
            parameters.append(param)
            self.params_name.append(key)
        self.params_shape = {
            name: backbone.state_dict()[name].shape for name in self.params_name
        }
        self.params_generator = nn.ParameterDict()
        for name, param in zip(self.params_name, parameters):
            self.params_generator[name.replace(".", "-")] = nn.Linear(
                self.args.pefll.hyper_hidden_dim, param.numel()
            )

    def forward(self, embedding):
        features = self.mlp(embedding)
        return OrderedDict(
            (
                name,
                self.params_generator[name.replace(".", "-")](features).reshape(
                    self.params_shape[name]
                ),
            )
            for name in self.params_name
        )
