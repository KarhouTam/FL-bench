"""
Differential Privacy Manager for FL-bench integration with Opacus.
Handles all DP-related operations in a centralized manner.
"""
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import torch

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier

if TYPE_CHECKING:
    from src.utils.logger import Logger


class DPManager:
    """
    Centralized manager for differential privacy operations.
    Handles model conversion, parameter mapping, and privacy tracking.
    """

    def __init__(self, args):
        self.args = args
        self.dp_config = args.dpfedavg

        # Logger will be injected by server after initialization
        self._logger: Optional["Logger"] = None

        # DP parameters
        self.epsilon = self.dp_config.epsilon
        self.delta = self.dp_config.delta
        self.max_grad_norm = self.dp_config.max_grad_norm
        self.noise_multiplier = getattr(self.dp_config, 'noise_multiplier', None)
        self.sample_rate = getattr(self.dp_config, 'sample_rate', -1.0)
        self.secure_mode = getattr(self.dp_config, 'secure_mode', False)
        self.rdp_alpha_max = getattr(self.dp_config, 'rdp_alpha_max', 64)

        # Privacy tracking
        self.privacy_stats = {
            "epsilon_spent": 0.0,
            "delta_spent": 0.0,
            "noise_multiplier": 0.0,
            "rounds_completed": 0,  # Track completed rounds instead of total steps
            "max_client_epsilon": 0.0,  # Track maximum client epsilon
        }

        # Per-client cumulative epsilon tracking
        self.client_epsilon_cumulative = {}  # {client_id: cumulative_epsilon}

    def set_logger(self, logger: "Logger"):
        """Inject server-side logger and output initialization message."""
        self._logger = logger
        self._log(
            f"🔒 DPManager initialized: Target privacy budget ε={self.epsilon}, δ={self.delta}\n"
            f"   Privacy tracking: Using max(client_epsilon) for parallel training\n"
            f"   Accounting: {getattr(self.dp_config, 'privacy_accountant', 'rdp').upper()} accountant"
        )

    def _log(self, msg: str):
        """Log message using server logger if available."""
        if self._logger:
            self._logger.log(msg)

    def _warn(self, msg: str):
        """Log warning using server logger if available."""
        if self._logger:
            self._logger.warn(msg)

    @staticmethod
    def create_privacy_stats_dict() -> Dict[str, Any]:
        """Create standardized privacy statistics dictionary structure for clients."""
        return {
            "epsilon_spent": 0.0,
            "delta_spent": 0.0,
            "noise_multiplier": 0.0,
            "steps_taken": 0,  # Client tracks its own processed steps
            "accountant_steps": 0  # Client tracks accountant's actual steps
        }

    def prepare_model_for_dp(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Prepare and convert model for DP training.
        Returns DP-compatible model.
        """
        # Check and fix model compatibility
        validator = ModuleValidator()
        if not validator.is_valid(model):
            try:
                fixed_model = validator.fix(model)
                self._log("Model automatically fixed for DP compatibility")
                return fixed_model
            except Exception:
                errors = validator.validate(model, strict=False)
                raise ValueError(f"Model incompatible with Opacus: {errors}")
        else:
            # Model is already compatible
            return model

    def get_dp_config_for_client(self, batch_size: int, dataset_size: int, local_epoch: int, trainloader_length: int) -> Dict[str, Any]:
        """Get DP configuration to send to clients with calculated sample_rate and noise_multiplier."""
        # Calculate effective sample_rate
        if self.sample_rate > 0:
            effective_sample_rate = min(0.99, self.sample_rate)
        else:
            effective_sample_rate = min(0.99, batch_size / dataset_size)

        # Calculate noise_multiplier if not provided
        if self.noise_multiplier is None:
            # Total training steps = epochs × batches per epoch
            steps = local_epoch * trainloader_length
            accountant = getattr(self.dp_config, 'privacy_accountant', 'rdp')
            effective_noise_multiplier = self.calculate_noise_multiplier(
                sample_rate=effective_sample_rate,
                steps=steps,
                accountant=accountant
            )
        else:
            effective_noise_multiplier = self.noise_multiplier

        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_grad_norm": self.max_grad_norm,
            "noise_multiplier": effective_noise_multiplier,
            "sample_rate": effective_sample_rate,
            "privacy_accountant": getattr(self.dp_config, 'privacy_accountant', 'rdp'),
            "secure_mode": self.secure_mode,
            "rdp_alpha_max": self.rdp_alpha_max
        }

    def update_privacy_stats(self, client_id: int, client_privacy_stats: Dict[str, Any]):
        """Update global privacy statistics from client reports with cumulative tracking.

        Args:
            client_id: The ID of the client reporting privacy stats
            client_privacy_stats: Privacy statistics from the client's current round
        """
        if client_privacy_stats:
            # Get epsilon from this round's training
            round_epsilon = client_privacy_stats.get("epsilon_spent", 0.0)

            # Accumulate epsilon for this client across rounds
            if client_id not in self.client_epsilon_cumulative:
                self.client_epsilon_cumulative[client_id] = 0.0

            self.client_epsilon_cumulative[client_id] += round_epsilon

            # Global epsilon is the maximum across all clients (parallel composition)
            max_epsilon = max(self.client_epsilon_cumulative.values()) if self.client_epsilon_cumulative else 0.0

            if max_epsilon > self.privacy_stats["epsilon_spent"]:
                self.privacy_stats["epsilon_spent"] = max_epsilon
                self.privacy_stats["max_client_epsilon"] = max_epsilon

    def increment_round(self):
        """Increment round counter after each training round."""
        self.privacy_stats["rounds_completed"] += 1

    def get_client_epsilon(self, client_id: int) -> float:
        """Get cumulative epsilon for a specific client.

        Args:
            client_id: The ID of the client

        Returns:
            float: Cumulative epsilon spent by this client across all rounds
        """
        return self.client_epsilon_cumulative.get(client_id, 0.0)

    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy report."""
        epsilon_remaining = max(0, self.epsilon - self.privacy_stats["epsilon_spent"])

        return {
            "target_epsilon": self.epsilon,
            "target_delta": self.delta,
            "epsilon_spent": self.privacy_stats["epsilon_spent"],
            "epsilon_remaining": epsilon_remaining,
            "privacy_exhausted": epsilon_remaining <= 0.01,  # Nearly exhausted
            "rounds_completed": self.privacy_stats["rounds_completed"],  # Track rounds instead of steps
            "noise_multiplier": self.privacy_stats["noise_multiplier"]
        }

    def validate_privacy_budget(self) -> Tuple[bool, str]:
        """Check if privacy budget is still available."""
        report = self.get_privacy_report()

        if report["privacy_exhausted"]:
            return False, f"Privacy budget exhausted. Used: {report['epsilon_spent']:.4f}/{self.epsilon}"

        return True, f"Privacy budget OK. Remaining: {report['epsilon_remaining']:.4f}"

    def calculate_noise_multiplier(
        self,
        sample_rate: float,
        steps: int,
        accountant: str = "rdp"
    ) -> float:
        """
        Calculate noise multiplier for given privacy parameters.

        Args:
            sample_rate: Sampling rate (batch_size / dataset_size)
            steps: Number of training steps
            accountant: Privacy accounting method ("rdp", "gdp", "prv")

        Returns:
            float: Calculated noise multiplier
        """
        if accountant == "rdp":
            return get_noise_multiplier(
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                sample_rate=sample_rate,
                steps=steps
            )
        else:
            # For other accountants, use RDP as fallback
            self._warn(f"Accountant {accountant} not fully implemented, using RDP")
            return get_noise_multiplier(
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                sample_rate=sample_rate,
                steps=steps
            )