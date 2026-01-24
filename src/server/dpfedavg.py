import logging
import math
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any, Dict

from src.client.dpfedavg import DPFedAvgClient
from src.server.fedavg import FedAvgServer
from src.utils.dp_manager import DPManager


class DPFedAvgServer(FedAvgServer):
    algorithm_name: str = "DP-FedAvg"
    all_model_params_personalized = False
    return_diff = False
    client_cls = DPFedAvgClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--epsilon", type=float, default=1.0,
                          help="Privacy budget parameter (smaller = more private)")
        parser.add_argument("--delta", type=float, default=1e-5,
                          help="Privacy parameter (should be < 1/dataset_size)")
        parser.add_argument("--max_grad_norm", type=float, default=1.0,
                          help="Maximum norm for gradient clipping")
        parser.add_argument("--noise_multiplier", type=float, default=None,
                          help="Noise multiplier (if None, will be auto-calculated)")
        parser.add_argument("--privacy_accountant", type=str, default="rdp", choices=["rdp", "gdp", "prv"],
                          help="Privacy accounting method")
        parser.add_argument("--sample_rate", type=float, default=-1.0,
                          help="Poisson sampling rate. If negative, calculated from batch_size/dataset_size")
        parser.add_argument("--secure_mode", type=bool, default=False,
                          help="Use cryptographically secure RNG (slower but more secure)")
        parser.add_argument("--rdp_alpha_max", type=int, default=64,
                          help="Maximum alpha for RDP accountant (larger = tighter bound)")
        return parser.parse_args(args_list)

    def __init__(self, args, init_trainer=True, init_model=True):
        # Initialize DP manager first
        self.dp_manager = DPManager(args)

        # Call parent initialization
        super().__init__(args, init_trainer=False, init_model=init_model)

        # Initialize trainer after DP setup
        if init_trainer:
            self.init_trainer()

        # Inject logger into DPManager now that parent init is complete
        self.dp_manager.set_logger(self.logger)

    def init_model(self, model=None, preprocess_func=None, postprocess_func=None):
        """Initialize model with DP preparation."""
        # First, initialize the model normally
        super().init_model(model, preprocess_func, postprocess_func)

        # Then prepare it for DP
        self.model = self.dp_manager.prepare_model_for_dp(self.model)

        # Use DP model parameters directly
        _init_global_params, _init_global_params_name = [], []
        for key, param in self.model.named_parameters():
            _init_global_params.append(param.data.clone())
            _init_global_params_name.append(key)

        self.public_model_param_names = _init_global_params_name
        self.public_model_params = OrderedDict(
            zip(_init_global_params_name, _init_global_params)
        )

    def package(self, client_id: int):
        """Package parameters for client with DP configuration."""
        base_package = super().package(client_id)

        # Get client dataset size and training parameters for DP calculation
        client_dataset_size = len(self.client_data_indices[client_id])
        batch_size = self.args.common.batch_size
        local_epoch = self.args.common.local_epoch

        # Calculate actual trainloader length (batches per epoch) for this client
        trainloader_length = math.ceil(client_dataset_size / batch_size)

        base_package["dp_config"] = self.dp_manager.get_dp_config_for_client(
            batch_size=batch_size,
            dataset_size=client_dataset_size,
            local_epoch=local_epoch,
            trainloader_length=trainloader_length
        )
        return base_package

    def get_client_model_params(self, client_id: int):
        """Get client model parameters."""
        # Get the model parameters (already in DP format)
        regular_params = self.public_model_params.copy()
        personal_params = self.clients_personal_model_params[client_id]

        return dict(
            regular_model_params=regular_params,
            personal_model_params=personal_params
        )

    def aggregate_client_updates(self, client_packages: OrderedDict[int, Dict[str, Any]]):
        """Aggregate client updates with DP parameter name handling."""
        # Update privacy statistics with client_id for cumulative tracking
        for client_id, package in client_packages.items():
            if "privacy_stats" in package:
                self.dp_manager.update_privacy_stats(client_id, package["privacy_stats"])

        # Check privacy budget
        budget_ok, budget_msg = self.dp_manager.validate_privacy_budget()
        if not budget_ok:
            logging.warning(f"Privacy budget warning: {budget_msg}")


        # Perform standard FedAvg aggregation directly
        super().aggregate_client_updates(client_packages)

        # Update the DP model with aggregated parameters (already in DP format)
        self.model.load_state_dict(self.public_model_params, strict=False)

        # Record completed training round
        self.dp_manager.increment_round()

    def display_metrics(self):
        """Display metrics including privacy information."""
        super().display_metrics()

        # Display privacy statistics with enhanced information
        if self.verbose:
            privacy_report = self.dp_manager.get_privacy_report()

            # Calculate percentage used
            epsilon_percentage = (privacy_report['epsilon_spent'] / privacy_report['target_epsilon']) * 100

            # Status indicator
            if epsilon_percentage >= 100:
                status_icon = "🔴"
                status_text = "BUDGET EXHAUSTED"
            elif epsilon_percentage >= 90:
                status_icon = "🟡"
                status_text = "WARNING"
            elif epsilon_percentage >= 50:
                status_icon = "🟢"
                status_text = "OK"
            else:
                status_icon = "🟢"
                status_text = "GOOD"

            # Get max client epsilon for additional info
            max_client_id = max(self.dp_manager.client_epsilon_cumulative.items(),
                              key=lambda x: x[1])[0] if self.dp_manager.client_epsilon_cumulative else None

            client_info = f", max_client={max_client_id}" if max_client_id is not None else ""

            self.logger.log(
                f"{status_icon} Privacy: ε={privacy_report['epsilon_spent']:.4f}/{privacy_report['target_epsilon']:.1f} "
                f"({epsilon_percentage:.1f}% used), "
                f"remaining={privacy_report['epsilon_remaining']:.4f}, "
                f"rounds={privacy_report['rounds_completed']}{client_info} [{status_text}]"
            )