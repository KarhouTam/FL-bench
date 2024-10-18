from collections import OrderedDict
from copy import deepcopy
from typing import Any, List

import torch

from src.server.fedavg import FedAvgServer
from src.client.svdfed import SVDFedClient
from src.utils.tools import NestedNamespace
from src.utils.my_utils import calculate_data_size
from src.utils.compressor_utils import CompressorCombin

setting_list = [
    'base.conv1.weight', 'base.conv1.bias', 'base.bn1.weight', 'base.bn1.bias', 
    'base.conv2.weight', 'base.conv2.bias', 'base.bn2.weight', 'base.bn2.bias', 
    'base.conv3.weight', 'base.conv3.bias', 'base.bn3.weight', 'base.bn3.bias', 
    'base.conv4.weight', 'base.conv4.bias', 'base.bn4.weight', 'base.bn4.bias', 
    'base.conv5.weight', 'base.conv5.bias', 'base.bn5.weight', 'base.bn5.bias', 
    'classifier.fc1.weight', 'classifier.fc1.bias', 'classifier.bn6.weight', 'classifier.bn6.bias', 
    'classifier.fc2.weight', 'classifier.fc2.bias', 'classifier.bn7.weight', 'classifier.bn7.bias', 
    'classifier.fc3.weight']

setting_dict = {key: (10,) for key in setting_list}

class SVDFedServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "SVDFed",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        
        self.init_trainer(SVDFedClient)
        print("Setting_dict: ", setting_dict)
        self.clients_compress_combin = {key: CompressorCombin(setting_dict, "SVDCompressor") for key in range(self.client_num)}
        self.server_compress_combin = CompressorCombin(setting_dict, "SVDCompressor")
        _ = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        # self.clients_combin_error = {key: deepcopy(_) for key in range(self.client_num)}
        
        self.combin_update_dict = {}
        self.full_grad = True
        
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
            compress_combin=self.clients_compress_combin[client_id],
            # combin_error = self.clients_combin_error[client_id],

            combin_update_dict = self.combin_update_dict,
            full_grad = self.full_grad
        )


    def unpack_client_model(self, packed_params):
        # self.logger.log(f"Unpacking client model {packed_params.keys()}")
        return self.server_compress_combin.uncompress(packed_params["combin_alpha"], self.model.state_dict())

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        def get_vector_by_list(vector_list: List[torch.Tensor]):
            return torch.stack([vector.flatten() for vector in vector_list], dim=0)
        
        self.logger.log(f"Round {self.current_epoch}")
        selected_clients = sorted(self.selected_clients)
        
        if self.current_epoch % 5 == 0:
            self.full_grad = True
        else:
            self.full_grad = False

        public_model_byte = calculate_data_size(self.public_model_params, set_sparse=self.set_sparse,set_layout=self.set_layout)

        for key, value in self.combin_update_dict.items():
            public_model_byte += calculate_data_size(value, set_sparse=self.set_sparse, set_layout=self.set_layout)

        clients_package = self.trainer.train()
        
        if self.full_grad:
            for client_id in selected_clients:
                self.clients_comm_recv_bytes[client_id] += public_model_byte
                byte = calculate_data_size(clients_package[client_id]['model_params_diff'], 
                                    set_sparse=self.set_sparse, 
                                    set_layout=self.set_layout)
                self.clients_comm_send_bytes[client_id] += byte

            self.aggregate(clients_package)

            self.logger.log(f"{clients_package[0]['model_params_diff'].keys()}")

            # clients_model_param_dict: {layer_name: [clent1_grad, client2_grad, ...]} in clients_package
            clients_model_param_dict = {
                key: get_vector_by_list([package["model_params_diff"][key] for package in clients_package.values()]) 
                for key in setting_dict
            }
            self.combin_update_dict = self.server_compress_combin.update_basis_by_vector(clients_model_param_dict)   

        else:
            for client_id, package in clients_package.items():
                self.clients_comm_recv_bytes[client_id] += public_model_byte
                byte = calculate_data_size(clients_package[client_id]['model_params_diff']["combin_alpha"], 
                                    set_sparse=self.set_sparse, 
                                    set_layout=self.set_layout)
                self.clients_comm_send_bytes[client_id] += byte
                
                package["model_params_diff"] = self.unpack_client_model(package["model_params_diff"])

            self.aggregate(clients_package)
            self.combin_update_dict = {}

