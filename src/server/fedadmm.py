from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict

import torch

from src.client.fedadmm import FedADMMClient
from src.server.fedavg import FedAvgServer


class FedADMMServer(FedAvgServer):
    algorithm_name: str = "FedADMM"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedADMMClient

    def __init__(self, *args, **kwargs):
        super(FedADMMServer, self).__init__(*args, **kwargs)
        # Initialize theta (global model parameters)
        self.theta = OrderedDict()
        for key, param in self.public_model_params.items():
            self.theta[key] = param.clone().to(self.device)
        # Current learning rate for global model update
        self.current_eta = self.args.fedadmm.eta

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        # FedADMM specific hyperparameters
        parser.add_argument("--rho", type=float, default=0.01,
                            help="Penalty parameter for ADMM")
        parser.add_argument("--fixed", type=int, default=0,
                            help="Fixed local epochs, 1 for fixed")
        parser.add_argument("--eta", type=float, default=1.0,
                            help="Learning rate of global model")
        parser.add_argument("--eta_2", type=float, default=0.5,
                            help="Learning rate of global model phase 2")
        parser.add_argument("--target_round", type=int, default=60,
                            help="The number of target round to change eta")
        return parser.parse_args(args_list)

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at
        server side) in each communication round."""
        # Check if we need to change the learning rate
        if self.current_epoch >= self.args.fedadmm.target_round:
            self.current_eta = self.args.fedadmm.eta_2

        # Train clients
        client_packages = self.trainer.train()
        
        # Aggregate client updates
        self.aggregate_client_updates(client_packages)

    @torch.no_grad()
    def aggregate_client_updates(self, client_packages: OrderedDict[int, Dict[str, Any]]):
        """Aggregate clients model parameters and produce global model
        parameters using FedADMM aggregation.

        Args:
            client_packages: Dict of client parameter packages
        """
        # Extract local_sum from client packages
        local_sums = [package["local_sum"] for package in client_packages.values()]
        
        # Calculate weights for weighted averaging
        client_weights = [package["weight"] for package in client_packages.values()]
        total_weight = sum(client_weights)
        
        # Average the local_sum values
        update_msg = {}
        for key in self.public_model_params.keys():
            if key in local_sums[0]:  # Check if key exists in local_sum
                # Stack all client updates for this parameter
                stacked_updates = torch.stack([
                    local_sum[key] * (client_weight / total_weight)
                    for local_sum, client_weight in zip(local_sums, client_weights)
                ], dim=0).to(self.device)
                
                # Sum along client dimension
                update_msg[key] = torch.sum(stacked_updates, dim=0)
        
        # Update theta using the learning rate
        for key in self.theta.keys():
            if key in update_msg:
                self.theta[key] = self.theta[key].to(self.device) + self.current_eta * update_msg[key]
        
        # Update global model parameters
        self.public_model_params = deepcopy(self.theta)
        self.model.load_state_dict(self.public_model_params, strict=False)
