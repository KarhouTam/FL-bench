from argparse import ArgumentParser, Namespace

import torch
from omegaconf import DictConfig

from src.client.fediir import FedIIRClient
from src.server.fedavg import FedAvgServer


class FedIIRServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--ema", type=float, default=0.95)
        parser.add_argument("--penalty", type=float, default=1e-3)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: DictConfig,
        algorithm_name: str = "FedIIR",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(
            args, algorithm_name, unique_model, use_fedavg_client_cls, return_diff
        )
        self.grad_mean = tuple(
            torch.zeros_like(p) for p in list(self.model.classifier.parameters())
        )
        self.calculating_grad_mean = False
        self.init_trainer(FedIIRClient)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["grad_mean"] = None
        if not self.calculating_grad_mean:
            server_package["grad_mean"] = self.calculate_grad_mean()

        return server_package

    def calculate_grad_mean(self):
        self.calculating_grad_mean = True
        batch_total = 0
        grad_sum = tuple(
            torch.zeros_like(p) for p in list(self.model.classifier.parameters())
        )
        client_packages = self.trainer.exec("grad", self.selected_clients)
        for client_id in self.selected_clients:
            batch_total += client_packages[client_id]["total_batch"]
            grad_sum = tuple(
                g1 + g2
                for g1, g2 in zip(grad_sum, client_packages[client_id]["grad_sum"])
            )
        grad_mean_new = tuple(grad / batch_total for grad in grad_sum)
        self.calculating_grad_mean = False
        return tuple(
            (self.args.fediir.ema * g1 + (1 - self.args.fediir.ema) * g2).cpu().clone()
            for g1, g2 in zip(self.grad_mean, grad_mean_new)
        )
