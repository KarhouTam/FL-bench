import pickle
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

from src.utils.tools import trainable_params, evalutate_model, Logger
from src.utils.models import DecoupledModel
from src.utils.constants import DATA_MEAN, DATA_STD
from data.utils.datasets import DATASETS


class FedAvgClient:
    def __init__(
        self,
        model: DecoupledModel,
        args: Namespace,
        logger: Logger,
        device: torch.device,
    ):
        self.args = args
        self.device = device
        self.client_id: int = None

        # load dataset and clients' data indices
        try:
            partition_path = PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")

        self.data_indices: List[List[int]] = partition["data_indices"]

        # --------- you can define your own data transformation strategy here ------------
        general_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[self.args.dataset], DATA_STD[self.args.dataset]
                )
            ]
            if self.args.dataset in DATA_MEAN and self.args.dataset in DATA_STD
            else []
        )
        general_target_transform = transforms.Compose([])
        train_data_transform = transforms.Compose([])
        train_target_transform = transforms.Compose([])
        # --------------------------------------------------------------------------------

        self.dataset = DATASETS[self.args.dataset](
            root=PROJECT_DIR / "data" / args.dataset,
            args=args.dataset_args,
            general_data_transform=general_data_transform,
            general_target_transform=general_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )

        self.trainloader: DataLoader = None
        self.valloader: DataLoader = None
        self.testloader: DataLoader = None
        self.trainset: Subset = Subset(self.dataset, indices=[])
        self.valset: Subset = Subset(self.dataset, indices=[])
        self.testset: Subset = Subset(self.dataset, indices=[])

        self.model = model.to(self.device)
        self.local_epoch = self.args.local_epoch
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
        if self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                params=trainable_params(self.model),
                lr=self.args.local_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params=trainable_params(self.model),
                lr=self.args.local_lr,
                weight_decay=self.args.weight_decay,
            )
        self.init_opt_state_dict = deepcopy(self.optimizer.state_dict())

    def load_dataset(self):
        """This function is for loading data indices for No.`self.client_id` client."""
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.testset.indices = self.data_indices[self.client_id]["test"]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.trainloader = DataLoader(self.trainset, self.args.batch_size, shuffle=True)
        self.valloader = DataLoader(self.valset, self.args.batch_size)
        self.testloader = DataLoader(self.testset, self.args.batch_size)

    def train_and_log(self, verbose=False) -> Dict[str, Dict[str, float]]:
        """This function includes the local training and logging process.

        Args:
            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: The logging info, which contains metric stats.
        """
        eval_results = {
            "before": {
                "train": {"loss": 0, "correct": 0, "size": 0},
                "val": {"loss": 0, "correct": 0, "size": 0},
                "test": {"loss": 0, "correct": 0, "size": 0},
            },
            "after": {
                "train": {"loss": 0, "correct": 0, "size": 0},
                "val": {"loss": 0, "correct": 0, "size": 0},
                "test": {"loss": 0, "correct": 0, "size": 0},
            },
        }
        eval_results["before"] = self.evaluate()
        if self.local_epoch > 0:
            self.fit()
            self.save_state()
            eval_results["after"] = self.evaluate()
        if verbose:
            colors = {"train": "yellow", "val": "green", "test": "cyan"}
            for split, flag, subset in [
                ["train", self.args.eval_train, self.trainset],
                ["val", self.args.eval_val, self.valset],
                ["test", self.args.eval_test, self.testset],
            ]:
                if len(subset) > 0 and flag:
                    self.logger.log(
                        "client [{}] [{}]({})  loss: {:.4f} -> {:.4f}   accuracy: {:.2f}% -> {:.2f}%".format(
                            self.client_id,
                            colors[split],
                            split,
                            eval_results["before"][split]["loss"]
                            / eval_results["before"][split]["size"],
                            eval_results["after"][split]["loss"]
                            / eval_results["after"][split]["size"],
                            eval_results["before"][split]["correct"]
                            / eval_results["before"][split]["size"]
                            * 100.0,
                            eval_results["after"][split]["correct"]
                            / eval_results["after"][split]["size"]
                            * 100.0,
                        )
                    )

        return eval_results

    def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):
        """Load model parameters received from the server.

        Args:
            new_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.
        """
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
        """Save client model personal parameters and the state of optimizer at the end of local training."""
        self.personal_params_dict[self.client_id] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (key in self.personal_params_name)
        }
        self.opt_state_dict[self.client_id] = deepcopy(self.optimizer.state_dict())

    def train(
        self,
        client_id: int,
        local_epoch: int,
        new_parameters: OrderedDict[str, torch.Tensor],
        return_diff=True,
        verbose=False,
    ) -> Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
        """
        The funtion for including all operations in client local training phase.
        If you wanna implement your method, consider to override this funciton.

        Args:
            client_id (int): The ID of client.

            local_epoch (int): The number of epochs for performing local training.

            new_parameters (OrderedDict[str, torch.Tensor]): Parameters of FL model.

            return_diff (bool, optional):
            Set as `True` to send the difference between FL model parameters that before and after training;
            Set as `False` to send FL model parameters without any change.  Defaults to True.

            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
            [The difference / all trainable parameters, the weight of this client, the evaluation metric stats].
        """
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(new_parameters)
        eval_results = self.train_and_log(verbose=verbose)

        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1

            return delta, len(self.trainset), eval_results
        else:
            return (
                trainable_params(self.model, detach=True),
                len(self.trainset),
                eval_results,
            )

    def fit(self):
        """
        The function for specifying operations in local training phase.
        If you wanna implement your method and your method has different local training operations to FedAvg, this method has to be overrided.
        """
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
    def evaluate(
        self, model: torch.nn.Module = None, force_eval=False
    ) -> Dict[str, float]:
        """The evaluation function. Would be activated before and after local training if `eval_test = True` or `eval_train = True`.

        Args:
            model (torch.nn.Module, optional): The target model needed evaluation (set to `None` for using `self.model`). Defaults to None.
            force_eval (bool, optional): Set as `True` when the server asking client to evaluate model.
        Returns:
            Dict[str, float]: The evaluation metric stats.
        """
        # disable train data transform while evaluating
        self.dataset.enable_train_transform = False

        target_model = self.model if model is None else model
        target_model.eval()
        train_loss, val_loss, test_loss = 0, 0, 0
        train_correct, val_correct, test_correct = 0, 0, 0
        train_size, val_size, test_size = 0, 0, 0
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if len(self.testset) > 0 and self.args.eval_test:
            test_loss, test_correct, test_size = evalutate_model(
                model=target_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.valset) > 0 and (force_eval or self.args.eval_val):
            val_loss, val_correct, val_size = evalutate_model(
                model=target_model,
                dataloader=self.valloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.trainset) > 0 and self.args.eval_train:
            train_loss, train_correct, train_size = evalutate_model(
                model=target_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )

        self.dataset.enable_train_transform = True
        return {
            "train": {
                "loss": train_loss,
                "correct": train_correct,
                "size": float(max(1, train_size)),
            },
            "val": {
                "loss": val_loss,
                "correct": val_correct,
                "size": float(max(1, val_size)),
            },
            "test": {
                "loss": test_loss,
                "correct": test_correct,
                "size": float(max(1, test_size)),
            },
        }

    def test(
        self, client_id: int, new_parameters: OrderedDict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Test function. Only be activated while in FL test round.

        Args:
            client_id (int): The ID of client.
            new_parameters (OrderedDict[str, torch.Tensor]): The FL model parameters.

        Returns:
            Dict[str, Dict[str, float]]: the evalutaion metrics stats.
        """
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)

        # set `size` as 1 for avoiding NaN.
        results = {
            "before": {
                "train": {"loss": 0, "correct": 0, "size": 1},
                "val": {"loss": 0, "correct": 0, "size": 1},
                "test": {"loss": 0, "correct": 0, "size": 1},
            },
            "after": {
                "train": {"loss": 0, "correct": 0, "size": 1},
                "val": {"loss": 0, "correct": 0, "size": 1},
                "test": {"loss": 0, "correct": 0, "size": 1},
            },
        }

        results["before"] = self.evaluate(force_eval=True)
        if self.args.finetune_epoch > 0:
            self.finetune()
            results["after"] = self.evaluate(force_eval=True)
        return results

    def finetune(self):
        """
        The fine-tune function. If your method has different fine-tuning opeation, consider to override this.
        This function will only be activated in FL test epoches.
        """
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
