import pickle
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from path import Path
from rich.console import Console
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize

PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()

from src.config.utils import trainable_params, evaluate, FLBenchOptimizer
from src.config.models import DecoupledModel
from data.utils.constants import MEAN, STD
from data.utils.datasets import DATASETS


class FedAvgClient:
    def __init__(self, model: DecoupledModel, args: Namespace, logger: Console):
        self.args = args
        self.device = torch.device(
            "cuda" if self.args.client_cuda and torch.cuda.is_available() else "cpu"
        )
        self.client_id: int = None

        # load dataset and clients' data indices
        try:
            partition_path = PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")

        self.data_indices: List[List[int]] = partition["data_indices"]

        # you can add more data transformation operations there manually
        transform = Compose(
            [Normalize(MEAN[self.args.dataset], STD[self.args.dataset])]
        )
        target_transform = None

        self.dataset = DATASETS[self.args.dataset](
            root=PROJECT_DIR / "data" / args.dataset,
            args=args.dataset_args,
            transform=transform,
            target_transform=target_transform,
        )

        self.trainloader: DataLoader = None
        self.testloader: DataLoader = None
        self.trainset: Subset = Subset(self.dataset, indices=[])
        self.testset: Subset = Subset(self.dataset, indices=[])

        self.model = model.to(self.device)
        self.local_epoch = self.args.local_epoch
        self.local_lr = self.args.local_lr
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.logger = logger
        self.personal_params_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.personal_params_name: List[str] = []
        self.init_personal_params_dict: Dict[str, torch.Tensor] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if not param.requires_grad
        }
        self.opt_state_dict = {}

        # you can change the type and arguments of optimizer and add more arguments there manually
        optimizer = torch.optim.SGD(
            trainable_params(self.model),
            self.local_lr,
            self.args.momentum,
            self.args.weight_decay,
        )
        scheduler = None
        if self.args.use_lr_scheduler:
            # you can change the type and arguments of lr scheduler there manually
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=50, gamma=0.9
            )
        self.optimizer = FLBenchOptimizer(optimizer=optimizer, scheduler=scheduler)
        self.init_opt_state_dict = deepcopy(self.optimizer.state_dict())

    def load_dataset(self):
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.testset.indices = self.data_indices[self.client_id]["test"]
        self.trainloader = DataLoader(self.trainset, self.args.batch_size)
        self.testloader = DataLoader(self.testset, self.args.batch_size)

    def train_and_log(self, verbose=False) -> Dict[str, Dict[str, float]]:
        before = {
            "train_loss": 0,
            "test_loss": 0,
            "train_correct": 0,
            "test_correct": 0,
            "train_size": 1,
            "test_size": 1,
        }
        after = deepcopy(before)
        before = self.evaluate()
        if self.local_epoch > 0:
            self.fit()
            self.save_state()
            after = self.evaluate()
        if verbose:
            if len(self.trainset) > 0 and self.args.eval_train:
                self.logger.log(
                    "client [{}] (train)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        before["train_loss"] / before["train_size"],
                        after["train_loss"] / after["train_size"],
                        before["train_correct"] / before["train_size"] * 100.0,
                        after["train_correct"] / after["train_size"] * 100.0,
                    )
                )
            if len(self.testset) > 0 and self.args.eval_test:
                self.logger.log(
                    "client [{}] (test)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        before["test_loss"] / before["test_size"],
                        after["test_loss"] / after["test_size"],
                        before["test_correct"] / before["test_size"] * 100.0,
                        after["test_correct"] / after["test_size"] * 100.0,
                    )
                )

        eval_stats = {"before": before, "after": after}
        return eval_stats

    def set_parameters(self, new_parameters: OrderedDict[str, torch.nn.Parameter]):
        personal_parameters = self.personal_params_dict.get(
            self.client_id, self.init_personal_params_dict
        )
        self.optimizer.load_state_dict(
            self.opt_state_dict.get(self.client_id, self.init_opt_state_dict)
        )
        self.model.load_state_dict(new_parameters, strict=False)
        # personal params would overlap the dummy params from new_parameters from the same layers
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
    ) -> Tuple[List[torch.nn.Parameter], int, Dict]:
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)
        eval_stats = self.train_and_log(verbose=verbose)

        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0.to(self.device) - p1

            return delta, len(self.trainset), eval_stats
        else:
            return (
                deepcopy(trainable_params(self.model)),
                len(self.trainset),
                eval_stats,
            )

    def fit(self):
        self.model.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
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
    def evaluate(self, model: torch.nn.Module = None) -> Dict[str, Dict[str, float]]:
        eval_model = self.model if model is None else model
        eval_model.eval()
        train_loss, test_loss = 0, 0
        train_correct, test_correct = 0, 0
        train_sample_num, test_sample_num = 0, 0
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if len(self.testset) > 0 and self.args.eval_test:
            test_loss, test_correct, test_sample_num = evaluate(
                model=eval_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.trainset) > 0 and self.args.eval_train:
            train_loss, train_correct, train_sample_num = evaluate(
                model=eval_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )

        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_correct": train_correct,
            "test_correct": test_correct,
            "train_size": float(max(1, train_sample_num)),
            "test_size": float(max(1, test_sample_num)),
        }

    def test(
        self, client_id: int, new_parameters: OrderedDict[str, torch.nn.Parameter]
    ):
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)

        before = {
            "train_loss": 0,
            "train_correct": 0,
            "train_size": 1.0,
            "test_loss": 0,
            "test_correct": 0,
            "test_size": 1.0,
        }
        after = deepcopy(before)

        before = self.evaluate()
        if self.args.finetune_epoch > 0:
            self.finetune()
            after = self.evaluate()
        return {"before": before, "after": after}

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
