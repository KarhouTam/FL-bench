"""
Simplified DP-FedAvg client that only handles DP training.
All parameter mapping and model compatibility is handled by the server.
"""
from copy import deepcopy
from typing import Any

import torch

from src.client.fedavg import FedAvgClient
from src.utils.metrics import Metrics
from opacus import PrivacyEngine
from src.utils.dp_manager import DPManager


class DPFedAvgClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)

        # DP state
        self.privacy_engine = None
        self.dp_config = None

        # Privacy tracking - use standardized structure
        self.privacy_stats = DPManager.create_privacy_stats_dict()

        # Store reference to original trainloader for DP wrapping
        self.original_trainloader = None

    def set_parameters(self, package: dict[str, Any]):
        """Set parameters and setup DP training."""
        self.dp_config = package["dp_config"]

        # Force reset Privacy Engine to avoid cross-client sharing
        self._reset_privacy_engine()

        # IMPORTANT: Load data indices BEFORE setting up DP
        # This ensures DPDataLoader is created with the correct dataset size
        super().set_parameters(package)

        # Save original trainloader before DP wrapping (to avoid recursive collate_fn wrapping)
        if self.original_trainloader is None:
            from torch.utils.data import DataLoader
            # Create a fresh DataLoader with same configuration as parent class
            self.original_trainloader = DataLoader(
                self.trainset,
                batch_size=self.args.common.batch_size,
                shuffle=True
            )

        # Now setup DP training with the correct trainloader
        self._setup_dp_training()

    def _reset_privacy_engine(self):
        """Reset Privacy Engine state to ensure client isolation"""
        if self.privacy_engine is not None:
            # Reset privacy engine for client isolation
            # (expected behavior each round, no warning needed)
            self.privacy_engine = None
        # Reset privacy_stats (noise_multiplier will be set again in setup)
        self.privacy_stats = DPManager.create_privacy_stats_dict()

    def _setup_dp_training(self):
        """Setup DP training with provided configuration."""
        # Note: Always reinitialize because _reset_privacy_engine() cleared old state

        # Use noise multiplier calculated by server
        noise_multiplier = self.dp_config["noise_multiplier"]

        # Initialize Privacy Engine with correct accountant and secure mode
        accountant_type = self.dp_config.get("privacy_accountant", "rdp")
        secure_mode = self.dp_config.get("secure_mode", False)

        # Create PrivacyEngine with string accountant type (Opacus API requirement)
        self.privacy_engine = PrivacyEngine(accountant=accountant_type, secure_mode=secure_mode)

        # Make model, optimizer, and data loader private
        # IMPORTANT: Use original_trainloader to avoid recursive collate_fn wrapping
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.original_trainloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=self.dp_config["max_grad_norm"],
        )

        self.privacy_stats["noise_multiplier"] = noise_multiplier

        # Update parameter names to match DP model (use underlying module to avoid _module prefix)
        underlying_model = self.model._module
        self.regular_params_name = list(key for key, _ in underlying_model.named_parameters())
        if self.args.common.buffers == "local":
            self.personal_params_name = [name for name, _ in underlying_model.named_buffers()]


    def fit(self):
        """DP training implementation using standard DPDataLoader iteration."""
        self.model.train()
        self.dataset.train()

        total_steps = 0

        # Standard training loop: iterate through DPDataLoader for each epoch
        for _ in range(self.local_epoch):
            epoch_loss = 0.0
            epoch_steps = 0

            # Use DPDataLoader iteration (Opacus handles Poisson sampling automatically)
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                logit = self.model(x)
                loss = self.criterion(logit, y)

                # Backward pass with DP
                # DPOptimizer automatically handles:
                # 1. Per-sample gradient computation
                # 2. Gradient clipping
                # 3. Noise addition
                # 4. Calling accountant.step()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1
                total_steps += 1

                # Update learning rate if scheduler is available
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        # Update steps_taken in privacy_stats (for client's own record)
        self.privacy_stats["steps_taken"] = total_steps

        # Update privacy statistics from accountant
        self._update_privacy_stats()


    def _update_privacy_stats(self):
        """Update privacy budget consumption statistics."""
        accountant = self.privacy_engine.accountant
        if len(accountant.history) > 0:
            epsilon_spent = accountant.get_epsilon(delta=self.dp_config["delta"])
            if not (epsilon_spent != epsilon_spent):  # Check for NaN
                self.privacy_stats["epsilon_spent"] = epsilon_spent
                self.privacy_stats["accountant_steps"] = len(accountant.history)

    def package(self):
        """Package client data including privacy statistics."""
        # Get parameters from the underlying DP wrapped model
        model_params = self.model._module.state_dict()

        # Create the package manually to handle DP model parameters
        client_package = dict[str, int | dict[str, dict[str, Metrics]]](
            weight=len(self.trainset),
            eval_results=self.eval_results,
            regular_model_params={
                key: model_params[key].clone().cpu() for key in self.regular_params_name
                if key in model_params
            },
            personal_model_params={
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
                if key in model_params
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
        )

        if self.return_diff:
            client_package["model_params_diff"] = {
                key: param_old - param_new
                for (key, param_new), param_old in zip[tuple[tuple[str, dict[str, Metrics]], Any]](
                    client_package["regular_model_params"].items(),
                    self.regular_model_params.values(),
                )
            }
            client_package.pop("regular_model_params")

        # Add privacy statistics
        client_package["privacy_stats"] = deepcopy(self.privacy_stats)

        return client_package

    def evaluate(self, model: torch.nn.Module = None) -> dict[str, Any]:
        """Evaluate model, handling DP-wrapped models."""
        target_model = self.model if model is None else model

        # Use underlying model from DP wrapper for evaluation
        unwrapped_model = target_model._module
        return super().evaluate(unwrapped_model)