# 梯度压缩
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch

from src.server.fedavg import FedAvgServer
from src.client.fedtest2 import FedTest2Client
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
    'classifier.fc1.bias' : (16, 8, 128),'classifier.bn6.weight' : (16, 8, 128),'classifier.bn6.bias' : (16, 8, 128),'classifier.bn6.running_mean' : (16, 8, 128),'classifier.bn6.running_var' : (16, 8, 128), # shape: [4096]

    'classifier.fc2.weight' : (64, 32, 512), # shape: [4096, 4096]
    'classifier.fc2.bias' : (2, 2, 64),'classifier.bn7.weight' : (2, 2, 64),'classifier.bn7.bias' : (2, 2, 64),'classifier.bn7.running_mean' : (2, 2, 64),'classifier.bn7.running_var' : (2, 2, 64), # shape: [4096]
    
    'classifier.fc3.weight' : (4, 4, 512),   # shape: [10, 4096]
    # # 'classifier.fc3.bias' : torch.Size([10]),
}

class FedTest2Server(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedTest2",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        
        self.init_trainer(FedTest2Client)
        print("Setting_dict: ", setting_dict)
        self.clients_compress_combin = {key: CompressorCombin(setting_dict, "SlideSVDCompressor") for key in range(self.client_num)}
        self.server_compress_combin = {key: CompressorCombin(setting_dict, "SlideSVDCompressor") for key in range(self.client_num)}
        _ = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
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
            total_weight=self.total_weight,
            client_id=client_id,
            local_epoch=self.clients_local_epoch[client_id],
            **self.get_client_model_params(client_id),
            optimizer_state=self.clients_optimizer_state[client_id],
            lr_scheduler_state=self.clients_lr_scheduler_state[client_id],
            return_diff=self.return_diff,
            compress_combin=self.clients_compress_combin[client_id],
            combin_error = self.clients_combin_error[client_id]
        )

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        selected_clients = sorted(self.selected_clients)
        
        public_model_byte = calculate_data_size(self.public_model_params, set_sparse=self.set_sparse,set_layout=self.set_layout)
        
        clients_package = self.trainer.train()

        for client_id in selected_clients:
            self.clients_comm_recv_bytes[client_id] += public_model_byte
            try:
                byte = calculate_data_size(clients_package[client_id]['model_params_diff']["combin_alpha"], 
                                        set_sparse=self.set_sparse, 
                                        set_layout=self.set_layout)
                for key, value in clients_package[client_id]['model_params_diff']["combin_update_dict"].items():
                    byte += calculate_data_size(value, set_sparse=self.set_sparse, set_layout=self.set_layout)
            except Exception as e:
                print(e, clients_package[client_id]['model_params_diff'].keys())
            self.clients_comm_send_bytes[client_id] += byte

        self.aggregate(clients_package)

    def unpack_client_model(self, client_id, packed_params):
        templete_model_params = self.model.state_dict()
        compress_combin = self.server_compress_combin[client_id]
        # combin_error = self.clients_combin_error[client_id]
        compress_combin.update(packed_params["combin_update_dict"])

        uncompress = compress_combin.uncompress(packed_params["combin_alpha"], templete_model_params)
        # for key, value in uncompress.items():
        #     value += combin_error[key]
        return uncompress
        # return packed_params

    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):
        for client_id, package in clients_package.items():
            package["model_params_diff"] = self.unpack_client_model(client_id, package["model_params_diff"])
        
        # old_global_model = deepcopy(self.public_model_params)
        self.total_weight = sum([package["weight"] for package in clients_package.values()])
        super().aggregate(clients_package)

        # for key, value in self.public_model_params.items():
        #     a = cos_similar(value, old_global_model[key])
        #     print(f"Layer: {key}, Similarity: {a:.4f}")