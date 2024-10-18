from collections import OrderedDict
from copy import deepcopy
import time
from typing import Any

import torch
from src.server.fedavg import FedAvgServer
from src.client.signsgd import SignSGDClient
from src.utils.tools import NestedNamespace
from src.utils.my_utils import calculate_data_size


class SignSGDServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "SignSGD",
        unique_model=True,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(SignSGDClient)
        self.public_model_sign_diffs = {}


    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        selected_clients = sorted(self.selected_clients)
        
        public_diff_byte = calculate_data_size(self.public_model_sign_diffs, set_sparse='all',set_layout='bit')

        clients_package = self.trainer.train()

        for client_id in selected_clients:
            self.clients_comm_recv_bytes[client_id] += public_diff_byte
            self.clients_personal_model_params[client_id] = clients_package[client_id]['personal_model_params']
            assert self.return_diff, "SignSGD should return diff"
            byte = calculate_data_size(clients_package[client_id]['model_params_diff'], set_sparse='all', set_layout='bit')
            self.clients_comm_send_bytes[client_id] += byte

        self.aggregate(clients_package)

    @torch.no_grad()
    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):
        """Aggregate clients model parameters and produce global model parameters.

        Args:
            clients_package: Dict of client parameter packages, with format:
            {
                `client_id`: {
                    `regular_model_params`: ...,
                    `optimizer_state`: ...,
                }
            }

            About the content of client parameter package, check `FedAvgClient.package()`.
        """
        clients_weight = [package["weight"] for package in clients_package.values()]
        weights = torch.tensor(clients_weight) / sum(clients_weight)
        if self.return_diff:  # inputs are model params diff
            for name, global_param in self.public_model_params.items():
                diffs = torch.stack(
                    [
                        package["model_params_diff"][name]
                        for package in clients_package.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(
                    diffs * weights.to(diffs.device), dim=-1, dtype=global_param.dtype
                ).to(global_param.device)
                # self.public_model_diffs[name].data -= aggregated
                self.public_model_sign_diffs[name] = torch.sign(aggregated)

    def package(self, client_id: int):
        """Package parameters that the client-side training needs.
        If you are implementing your own FL method and your method has different parameters to FedAvg's
        that passes from server-side to client-side, this method need to be overrided.
        All this method should do is returning a dict that contains all parameters.

        Args:
            client_id: The client ID.

        Returns:
            A dict of parameters: {
                `client_id`: The client ID.
                `local_epoch`: The num of epoches that client local training performs.
                `client_model_params`: The client model parameter dict.
                `optimizer_state`: The client model optimizer's state dict.
                `lr_scheduler_state`: The client learning scheduler's state dict.
                `return_diff`: Flag that indicates whether client should send parameters difference.
                    `False`: Client sends vanilla model parameters;
                    `True`: Client sends `diff = global - local`.
            }.
        """
        return dict(
            client_id=client_id,
            local_epoch=self.clients_local_epoch[client_id],
            **self.get_client_model_params(client_id),
            optimizer_state=self.clients_optimizer_state[client_id],
            lr_scheduler_state=self.clients_lr_scheduler_state[client_id],
            return_diff=self.return_diff,
            public_model_diffs=self.public_model_sign_diffs,
        )  
