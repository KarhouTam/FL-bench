from collections import OrderedDict
from copy import deepcopy
from typing import Any
import torch
from src.client.fedavg import FedAvgClient


def get_model_sign_gradient(model):
    """
    Description:
        - get gradients from model, and store in a OrderDict
    
    Args:
        - model: (torch.nn.Module), torch model
    
    Returns:
        - grads in OrderDict
    """
    grads = OrderedDict()
    for name, params in model.named_parameters():
        grad = params.grad
        if grad is not None:
            grads[name] = torch.sign(grad)
    return grads


class SignSGDClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.model_params_sign_diff = {}
        self.old_model_params = {}
    
    def set_parameters(self, package: dict[str, Any]):
        _ = super().set_parameters(package)
        public_model_diffs = package["public_model_diffs"]

        for name, params in self.model.named_parameters():
            if name in public_model_diffs:
                params.grad = public_model_diffs[name].to(params.grad.device)
        self.optimizer.step()
        self.old_model_params = deepcopy(self.model.state_dict())


    def fit(self):
        self.model.train()
        self.dataset.train()
        
        self.optimizer.zero_grad()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                loss.backward()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()


    def package(self):
        """Package data that client needs to transmit to the server.
        You can override this function and add more parameters.

        Returns:
            A dict: {
                `weight`: Client weight. Defaults to the size of client training set.
                `regular_model_params`: Client model parameters that will join parameter aggregation.
                `model_params_diff`: The parameter difference between the client trained and the global. `diff = global - trained`.
                `eval_results`: Client model evaluation results.
                `personal_model_params`: Client model parameters that absent to parameter aggregation.
                `optimzier_state`: Client optimizer's state dict.
                `lr_scheduler_state`: Client learning rate scheduler's state dict.
            }
        """
        self.model
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            personal_model_params=self.old_model_params,
            # personal_model_params=deepcopy(self.model.state_dict()),
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
            # model_params_diff = {
            #     key: torch.sign(param_old - param_new) 
            #     for (key, param_new), param_old in zip(
            #         self.model.state_dict().items(),
            #         self.old_model_params.values(),
            #     )
            # }
            model_params_diff = get_model_sign_gradient(self.model)
        )
        return client_package

