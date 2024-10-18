from collections import OrderedDict
from typing import Any

import torch
from src.server.fedavg import FedAvgServer
from src.client.fedpaq import FedPAQClient
from src.utils.tools import NestedNamespace
from src.utils.my_utils import QSGDQuantizer, calculate_data_size




class FedPAQServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedPAQ",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(FedPAQClient)
        self.quantizer = QSGDQuantizer(16)

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        selected_clients = sorted(self.selected_clients)
        
        public_model_byte = calculate_data_size(self.public_model_params, set_sparse=self.set_sparse,set_layout=self.set_layout)
        
        clients_package = self.trainer.train()

        for client_id in selected_clients:
            self.clients_comm_recv_bytes[client_id] += public_model_byte
            assert self.return_diff, "The return_diff must be True in FedPAQ."
            try:
                byte = calculate_data_size(clients_package[client_id]['model_params_diff']["tensors"], 
                                        set_sparse=self.set_sparse, 
                                        set_layout=self.set_layout)
                byte += calculate_data_size(clients_package[client_id]['model_params_diff']["scales"], 
                        set_sparse=self.set_sparse, 
                        set_layout=self.set_layout)
            except:
                print(clients_package[client_id]['model_params_diff'])
            self.clients_comm_send_bytes[client_id] += byte

        self.aggregate(clients_package)

    def unpack_client_model(self, packed_model):
        quantized_model = packed_model["tensors"]
        scale = packed_model["scales"]
        dequantized_model = {}

        for key in quantized_model.keys():
            if scale[key] == 0:
                dequantized_model[key] = quantized_model[key].clone().detach()
            else:
                dequantized_model[key] = self.quantizer.dequantize(quantized_model[key], 0, scale[key])
        return dequantized_model

    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):
        for client_id, package in clients_package.items():
            package["model_params_diff"] = self.unpack_client_model(package["model_params_diff"])
        
        super().aggregate(clients_package)