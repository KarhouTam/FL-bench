from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch

from src.server.fedavg import FedAvgServer
from src.client.fedtest5 import FedTest5Client
from src.utils.tools import NestedNamespace
from src.utils.my_utils import calculate_data_size
from src.utils.compressor_utils import CompressorCombin

setting_dict = {
    # 'base.conv1.weight' : (2, 2, 5),    # shape: [64, 3, 5, 5]
    # 'base.conv1.bias' : (2, 1, 8),'base.bn1.weight' : (2, 1, 8),'base.bn1.bias' : (2, 1, 8),'base.bn1.running_mean' : (2, 1, 8),'base.bn1.running_var' : (2, 1, 8),  # shape: [64]

    # 'base.conv2.weight' : (4, 4, 25),  # shape: [192, 64, 5, 5]
    # 'base.conv2.bias' : (5, 2, 32),'base.bn2.weight' : (5, 2, 32),'base.bn2.bias' : (5, 2, 32),'base.bn2.running_mean' : (5, 2, 32),'base.bn2.running_var' : (5, 2, 32), # shape: [192]

    # 'base.conv3.weight' : (8, 8, 36), # shape: [384, 192, 3, 3]
    # 'base.conv3.bias' : (6, 3, 16),'base.bn3.weight' : (6, 3, 16),'base.bn3.bias' : (6, 3, 16),'base.bn3.running_mean' : (6, 3, 16),'base.bn3.running_var' : (6, 3, 16), # shape: [384]

    # 'base.conv4.weight' : (8, 8, 36), # shape: [256, 384, 3, 3]
    # 'base.conv4.bias' : (4, 2, 16),'base.bn4.weight' : (4, 2, 16),  'base.bn4.bias' : (4, 2, 16),'base.bn4.running_mean' : (4, 2, 16),'base.bn4.running_var' : (4, 2, 16),  # shape: [256]

    # 'base.conv5.weight' : (8, 8, 36), # shape: [256, 256, 3, 3]
    # 'base.conv5.bias' : (4, 2, 16),'base.bn5.weight' : (4, 2, 16),'base.bn5.bias' : (4, 2, 16),'base.bn5.running_mean' : (4, 2, 16),'base.bn5.running_var' : (4, 2, 16),  # shape: [256]

    'classifier.fc1.weight' : (64, 32, 512), # shape: [4096, 9216]
    'classifier.fc1.bias' : (16, 8, 128),'classifier.bn6.weight' : (16, 8, 128),'classifier.bn6.bias' : (16, 8, 128), # shape: [4096]

    'classifier.fc2.weight' : (64, 32, 512), # shape: [4096, 4096]
    'classifier.fc2.bias' : (2, 2, 64),'classifier.bn7.weight' : (2, 2, 64),'classifier.bn7.bias' : (2, 2, 64), # shape: [4096]
    
    'classifier.fc3.weight' : (4, 4, 512),   # shape: [10, 4096]
    # # 'classifier.fc3.bias' : torch.Size([10]),
}


class FedTest5Server(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedTest5",
        unique_model=True,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.logger.log("Begin to initialize FedTest5Server.")
        self.init_trainer(FedTest5Client)
        print("Setting_dict: ", setting_dict)
        
        # 用于broadcast的
        self.clients_broadcast_compress_combin = {key: CompressorCombin(setting_dict, "SlideSVDCompressor") for key in range(self.client_num)}
        self.server_broadcast_compress_combin = CompressorCombin(setting_dict, "SlideSVDCompressor")
        self.clients_last_global_model_params = {key: deepcopy(self.model.state_dict()) for key in range(self.client_num)}
        _ = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.server_combin_error = _
        self.aggregated_grad = {}
        self.global_grad = {'combin_alpha':{}, 'combin_update_dict':{}}

        self.test_model = deepcopy(self.model)

        # 用于upload的
        self.clients_upload_compress_combin = {key: CompressorCombin(setting_dict, "SlideSVDCompressor") for key in range(self.client_num)}
        self.server_upload_compress_combin = {key: CompressorCombin(setting_dict, "SlideSVDCompressor") for key in range(self.client_num)}
        self.clients_combin_error = {key: deepcopy(_) for key in range(self.client_num)}
        self.total_weight = None


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

            # 用于broadcast的
            last_global_model_params=self.clients_last_global_model_params[client_id],
            global_grad=self.global_grad,
            broadcast_compress_combin=self.clients_broadcast_compress_combin[client_id],
            test_model = self.test_model,

            # 用于upload的
            total_weight=self.total_weight,
            upload_compress_combin=self.clients_upload_compress_combin[client_id],
            combin_error = self.clients_combin_error[client_id]
        )

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        selected_clients = sorted(self.selected_clients)

        # 用于broadcast的
        if self.global_grad['combin_alpha'] != {}:
            recv_byte = calculate_data_size(self.global_grad['combin_alpha'], set_sparse=self.set_sparse, set_layout=self.set_layout)
            for key, value in self.global_grad['combin_update_dict'].items():
                recv_byte += calculate_data_size(value, set_sparse=self.set_sparse, set_layout=self.set_layout)
        else:
            recv_byte = calculate_data_size(self.public_model_params, set_sparse=self.set_sparse, set_layout=self.set_layout)

        clients_package = self.trainer.train()
        
        for client_id in selected_clients:
            self.clients_comm_recv_bytes[client_id] += recv_byte
            # 用于broadcast的
            self.clients_personal_model_params[client_id] = clients_package[client_id]['personal_model_params']
            
            # 用于upload的
            try:
                byte = calculate_data_size(clients_package[client_id]['model_params_diff']["combin_alpha"], 
                                        set_sparse=self.set_sparse, 
                                        set_layout=self.set_layout)
                for key, value in clients_package[client_id]['model_params_diff']["combin_update_dict"].items():
                    byte += calculate_data_size(value, set_sparse=self.set_sparse, set_layout=self.set_layout)
            except Exception as e:
                print(e)
                print(clients_package[client_id]['model_params_diff'].keys())

            self.clients_comm_send_bytes[client_id] += byte

        self.aggregate(clients_package)

        # 用于broadcast的
        combin_alpha, combin_update_dict, combin_error = self.server_broadcast_compress_combin.compress(self.aggregated_grad, can_update_basis_func=lambda: True)
        self.server_combin_error = combin_error
        self.global_grad['combin_alpha'] = combin_alpha
        self.global_grad['combin_update_dict'] = combin_update_dict


    # 用于upload的
    def unpack_client_model(self, client_id, packed_params):
        templete_model_params = self.model.state_dict()
        compress_upload_combin = self.server_upload_compress_combin[client_id]
        compress_upload_combin.update(packed_params["combin_update_dict"])
        uncompress = compress_upload_combin.uncompress(packed_params["combin_alpha"], templete_model_params)
        return uncompress


    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):
        # 用于upload的
        for client_id, package in clients_package.items():
            package["model_params_diff"] = self.unpack_client_model(client_id, package["model_params_diff"])
        
        clients_weight = [package["weight"] for package in clients_package.values()]
        
        # 用于upload的
        self.total_weight = sum(clients_weight)
        
        # 用于broadcast的
        weights = torch.tensor(clients_weight) / sum(clients_weight)
        for name, global_param in self.public_model_params.items():
            diffs = torch.stack(
                [
                    package["model_params_diff"][name]
                    for package in clients_package.values()
                ],
                dim=-1,
            )
            self.aggregated_grad[name] = torch.sum(
                diffs * weights, dim=-1, dtype=global_param.dtype
            ).to(global_param.device)
            self.public_model_params[name].data -= self.aggregated_grad[name]
            self.public_model_params[name].data += self.server_combin_error[name].to(self.public_model_params[name].device)
            error_norm = self.server_combin_error[name].norm()
            self.logger.log(f"Layer {name} error_norm: {error_norm}")