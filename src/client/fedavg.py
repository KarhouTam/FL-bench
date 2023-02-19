import sys
import pickle
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple

import torch
from path import Path
from rich.console import Console
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize

_CURRENT_DIR = Path(__file__).parent.abspath()
_PROJECT_DIR = _CURRENT_DIR.parent.parent.abspath()

sys.path.append(_PROJECT_DIR)
sys.path.append(_PROJECT_DIR / "src")
sys.path.append(_PROJECT_DIR / "data" / "utils")

from config.utils import trainable_params
from data.utils.constants import MEAN, STD
from data.utils.datasets import DATASETS


class FedAvgClient:
    def __init__(self, model: torch.nn.Module, args: Namespace, logger: Console):
        self.args = args
        self.device = torch.device(
            "cuda" if self.args.client_cuda and torch.cuda.is_available() else "cpu"
        )
        self.client_id: int = None

        # load dataset and clients' data indices
        try:
            partition_path = _PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")

        self.data_indices = partition["data_indices"]

        transform = Compose(
            [Normalize(MEAN[self.args.dataset], STD[self.args.dataset])]
        )
        # transform = None
        target_transform = None

        self.dataset = DATASETS[self.args.dataset](
            root=_PROJECT_DIR / "data" / args.dataset,
            args=args.dataset_args,
            transform=transform,
            target_transform=target_transform,
        )

        self.trainloader: DataLoader = None
        self.testloader: DataLoader = None
        self.trainset: Subset = None
        self.testset: Subset = None

        self.model = model.to(self.device)
        self.local_epoch = self.args.local_epoch
        self.local_lr = self.args.local_lr
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.logger = logger
        self.personal_params_dict: Dict[str, Dict[str, torch.Tensor]] = {}
        self.personal_params_name: List[str] = []
        self.init_personal_params_dict = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if not param.requires_grad
        }
        self.opt_state_dict = {}
        self.optimizer = SGD(
            trainable_params(self.model),
            self.local_lr,
            self.args.momentum,
            self.args.weight_decay,
        )

    def load_dataset(self):
        idx_train, idx_test = (
            self.data_indices[self.client_id]["train"],
            self.data_indices[self.client_id]["test"],
        )
        self.trainset = Subset(self.dataset, indices=idx_train)
        self.testset = Subset(self.dataset, indices=idx_test)
        self.trainloader = DataLoader(self.trainset, self.args.batch_size)
        self.testloader = DataLoader(self.testset, self.args.batch_size)

    def train_and_log(self, verbose=False):
        loss_before, loss_after = 0, 0
        correct_before, correct_after = 0, 0
        num_samples = 1
        if self.args.eval:
            loss_before, correct_before, num_samples = self.evaluate()
        if self.local_epoch > 0:
            self.fit()
            self.save_state()
            if self.args.eval:
                loss_after, correct_after, _ = self.evaluate()
        if verbose:
            self.logger.log(
                "client [{}]  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                    self.client_id,
                    loss_before / num_samples,
                    loss_after / num_samples,
                    correct_before / num_samples * 100.0,
                    correct_after / num_samples * 100.0,
                )
            )

        eval_stats = {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "correct_before": correct_before,
            "correct_after": correct_after,
            "num_samples": num_samples,
        }
        return eval_stats

    def set_parameters(self, new_parameters: OrderedDict[str, torch.nn.Parameter]):
        personal_parameters = self.init_personal_params_dict
        if self.client_id in self.personal_params_dict.keys():
            personal_parameters = self.personal_params_dict[self.client_id]
        if self.client_id in self.opt_state_dict.keys():
            self.optimizer.load_state_dict(self.opt_state_dict[self.client_id])
        self.model.load_state_dict(new_parameters, strict=False)
        # personal params would overlap the dummy params from new_parameters at the same layers
        self.model.load_state_dict(personal_parameters, strict=False)

    def save_state(self):
        self.personal_params_dict[self.client_id] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (key in self.personal_params_name)
        }
        self.opt_state_dict[self.client_id] = deepcopy(self.optimizer.state_dict())

    def train(
        self,
        client_id: int,
        new_parameters: OrderedDict[str, torch.nn.Parameter],
        return_diff=True,
        verbose=False,
    ):
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)
        stats = self.train_and_log(verbose=verbose)

        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0.to(self.device) - p1

            return delta, len(self.trainloader.dataset), stats
        else:
            return (
                deepcopy(trainable_params(self.model)),
                len(self.trainloader.dataset),
                stats,
            )

    def fit(self):
        self.model.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # when the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, int]:
        self.model.eval()
        loss = 0
        correct = 0
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        for x, y in self.testloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss += criterion(logits, y)
            pred = torch.argmax(logits, -1)
            correct += (pred == y).int().sum()
        return loss.item(), correct.item(), len(self.testset)

    def test(self, client_id, new_parameters):
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)
        loss_before, loss_after = 0, 0
        correct_before, correct_after = 0, 0
        loss_before, correct_before, num_samples = self.evaluate()
        if self.args.finetune_epoch > 0:
            self.finetune()
            loss_after, correct_after, _ = self.evaluate()
        eval_stats = {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "correct_before": correct_before,
            "correct_after": correct_after,
            "num_samples": num_samples,
        }
        return eval_stats

    def finetune(self):
        self.model.train()
        for _ in range(self.args.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
