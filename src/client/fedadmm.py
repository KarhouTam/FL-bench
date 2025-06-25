from collections import OrderedDict
from copy import deepcopy
import torch
from src.client.fedavg import FedAvgClient


class FedADMMClient(FedAvgClient):
    def __init__(self, **commons):
        super(FedADMMClient, self).__init__(**commons)
        # Initialize alpha (Lagrangian multipliers)
        self.alpha = {}
        weights = deepcopy(self.model.state_dict())
        for key in weights.keys():
            self.alpha[key] = torch.zeros_like(weights[key]).to(self.device)
        
        # Store previous model parameters and alpha for update calculation
        self.model_prev = None
        self.alpha_prev = None
        
        # Store theta (global model parameters) for local update
        self.theta = None

    def set_parameters(self, package):
        super().set_parameters(package)
        # Store theta (global model parameters)
        self.theta = deepcopy(OrderedDict(
            (key, param.clone().to(self.device))
            for key, param in self.model.state_dict().items()
            if key in self.regular_params_name
        ))
        
        # Store previous model parameters and alpha for update calculation
        self.model_prev = deepcopy(self.model.state_dict())
        self.alpha_prev = deepcopy(self.alpha)

    def fit(self):
        self.model.train()
        self.dataset.train()
        
        # Get the number of local epochs
        if self.args.fedadmm.fixed == 1:
            local_epochs = self.local_epoch
        else:
            # Random number of local epochs between 1 and local_epoch
            local_epochs = torch.randint(1, self.local_epoch + 1, (1,), device=self.device).item()
        
        for _ in range(local_epochs):
            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logit = self.model(x)
                loss = self.criterion(logit, y)
                loss.backward()
                
                # Get current model weights before optimization step
                model_weights_pre = deepcopy(self.model.state_dict())
                
                # Add ADMM regularization term to gradients
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in self.regular_params_name:
                        # Add Lagrangian term and proximal term to gradient
                        param.grad = param.grad + (self.alpha[name] + 
                                                  self.args.fedadmm.rho * 
                                                  (model_weights_pre[name].to(self.device) - self.theta[name].to(self.device)))
                
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
        # Update alpha (Lagrangian multipliers) after local training
        weights = self.model.state_dict()
        for key in self.alpha.keys():
            if key in self.regular_params_name:
                self.alpha[key] = self.alpha[key] + self.args.fedadmm.rho * (weights[key].to(self.device) - self.theta[key].to(self.device))

    def package(self):
        """Package data that client needs to transmit to the server.
        
        Returns:
            A dict with client updates including local_sum for FedADMM aggregation.
        """
        base_package = super().package()
        
        # Calculate local_sum for FedADMM
        weights = self.model.state_dict()
        local_sum = {}
        
        for key in self.regular_params_name:
            if key in weights and key in self.model_prev and key in self.alpha and key in self.alpha_prev:
                local_sum[key] = (weights[key].to(self.device) - self.model_prev[key].to(self.device)) + (1/self.args.fedadmm.rho) * (self.alpha[key] - self.alpha_prev[key])
        
        # Add local_sum to the package
        base_package['local_sum'] = local_sum
        
        return base_package
