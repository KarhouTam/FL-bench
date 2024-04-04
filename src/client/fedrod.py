from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn.functional as F

from fedavg import FedAvgClient
from src.utils.models import DecoupledModel
from src.utils.tools import Logger, NestedNamespace, count_labels, trainable_params


def balanced_softmax_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    label_counts: torch.Tensor,
):
    logits = logits + (label_counts**gamma).unsqueeze(0).expand(logits.shape).log()
    loss = F.cross_entropy(logits, targets, reduction="mean")
    return loss


class FedRoDClient(FedAvgClient):
    def __init__(
        self,
        model: DecoupledModel,
        hypernetwork: torch.nn.Module,
        args: NestedNamespace,
        logger: Logger,
        device: torch.device,
    ):
        super().__init__(FedRoDModel(model, args.fedrod.eval_per), args, logger, device)
        self.hypernetwork: torch.nn.Module = None
        self.hyper_optimizer = None
        if self.args.fedrod.hyper:
            self.hypernetwork = hypernetwork.to(self.device)
            self.hyper_optimizer = torch.optim.SGD(
                trainable_params(self.hypernetwork), lr=self.args.fedrod.hyper_lr
            )
        self.personal_params_name.extend(
            [key for key, _ in self.model.named_parameters() if "personalized" in key]
        )

    def set_parameters(self, new_generic_parameters: OrderedDict[str, torch.Tensor]):
        personal_parameters = self.personal_params_dict.get(
            self.client_id, self.init_personal_params_dict
        )
        self.optimizer.load_state_dict(
            self.opt_state_dict.get(self.client_id, self.init_opt_state_dict)
        )
        self.model.generic_model.load_state_dict(new_generic_parameters, strict=False)
        self.model.load_state_dict(personal_parameters, strict=False)

    def train(
        self,
        client_id: int,
        local_epoch: int,
        new_parameters: OrderedDict[str, torch.Tensor],
        hyper_parameters: OrderedDict[str, torch.Tensor],
        return_diff=False,
        verbose=False,
    ):
        self.client_id = client_id
        if self.args.fedrod.hyper:
            self.hypernetwork.load_state_dict(hyper_parameters, strict=False)
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(new_parameters)
        eval_results = self.train_and_log(verbose=verbose)

        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                new_parameters.items(), trainable_params(self.model.generic_model)
            ):
                delta[name] = p0 - p1

            hyper_delta = None
            if self.args.fedrod.hyper:
                hyper_delta = OrderedDict()
                for (name, p0), p1 in zip(
                    hyper_parameters.items(), trainable_params(self.hypernetwork)
                ):
                    hyper_delta[name] = p0 - p1

            return delta, hyper_delta, len(self.trainset), eval_results
        else:
            return (
                trainable_params(self.model.generic_model, detach=True),
                trainable_params(self.hypernetwork, detach=True),
                len(self.trainset),
                eval_results,
            )

    def fit(self):
        self.model.train()
        self.dataset.train()
        label_counts = torch.tensor(
            count_labels(self.dataset, self.trainset.indices), device=self.device
        )
        # if using hypernetwork for generating personalized classifier parameters and client is first-time selected
        if self.args.fedrod.hyper and self.client_id not in self.personal_params_dict:
            label_distrib = label_counts / label_counts.sum()
            classifier_params = self.hypernetwork(label_distrib)
            clf_weight_numel = self.model.generic_model.classifier.weight.numel()
            self.model.personalized_classifier.weight.data = (
                classifier_params[:clf_weight_numel]
                .reshape(self.model.personalized_classifier.weight.shape)
                .detach()
                .clone()
            )
            self.model.personalized_classifier.bias.data = (
                classifier_params[clf_weight_numel:]
                .reshape(self.model.personalized_classifier.bias.shape)
                .detach()
                .clone()
            )

        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit_g, logit_p = self.model(x)
                loss_g = balanced_softmax_loss(
                    logit_g, y, self.args.fedrod.gamma, label_counts
                )
                loss_p = self.criterion(logit_p, y)
                loss = loss_g + loss_p
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.args.fedrod.hyper and self.client_id not in self.personal_params_dict:
            # This part has no references on the FedRoD paper
            trained_classifier_params = torch.cat(
                [
                    torch.flatten(self.model.personalized_classifier.weight.data),
                    torch.flatten(self.model.personalized_classifier.bias.data),
                ]
            )
            hyper_loss = F.mse_loss(
                classifier_params, trained_classifier_params, reduction="sum"
            )
            self.hyper_optimizer.zero_grad()
            hyper_loss.backward()
            self.hyper_optimizer.step()

    def finetune(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                if self.args.fedrod.eval_per:
                    _, logit_p = self.model(x)
                    loss = self.criterion(logit_p, y)
                else:
                    logit_g, _ = self.model(x)
                    loss = self.criterion(logit_g, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


class FedRoDModel(DecoupledModel):
    def __init__(self, generic_model: DecoupledModel, eval_per):
        super().__init__()
        self.generic_model = generic_model
        self.personalized_classifier = deepcopy(generic_model.classifier)
        self.eval_per = eval_per

    def forward(self, x):
        z = torch.relu(self.generic_model.get_final_features(x, detach=False))
        logit_g = self.generic_model.classifier(z)
        logit_p = self.personalized_classifier(z)
        if self.training:
            return logit_g, logit_p
        else:
            if self.eval_per:
                return logit_p
            else:
                return logit_g
