from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import torch

from src.server.fedavg import FedAvgServer
from src.client.svdfed import SVDFedClient
from src.utils.tools import NestedNamespace
from src.utils.my_utils import calculate_data_size
from src.utils.compressor_utils import CompressorCombin

class SVDFedServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--setting_dict", type=dict)
        parser.add_argument("--kp", type=float, default=1)
        parser.add_argument("--ki", type=float, default=1)
        parser.add_argument("--kd", type=float, default=1)
        parser.add_argument("--gamma", type=float, default=18)
        parser.add_argument("--fixed_adj_freq", type=int, default=0, help="Fixed adjust frequency, if 0, adjust dynamically")
        return parser.parse_args(args=args_list)
    
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

        if args.svdfed.setting_dict is None:
            setting_dict = {}
        else:
            setting_dict = args.svdfed.setting_dict.to_dict()
        
        self.fixed_adj_freq = args.svdfed.fixed_adj_freq

        self.kp, self.ki, self.kd = args.svdfed.kp, args.svdfed.ki, args.svdfed.kd
        self.gamma = args.svdfed.gamma

        self.server_R = 6
        print("Setting_dict: ", setting_dict)
        self.clients_compress_combin = {key: CompressorCombin(setting_dict, "SVDCompressor") for key in range(self.client_num)}
        self.server_compress_combin = CompressorCombin(setting_dict, "SVDCompressor")
        _ = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        # self.clients_combin_error = {key: deepcopy(_) for key in range(self.client_num)}
        
        self.combin_update_dict = {}
        self.request_full_grad_params_name = list(self.server_compress_combin.setting_dict.keys())
        
        self.error_dict = {}
        self.total_error_dict = {}
        self.Z_thr = {}

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
            
            combin_update_dict = self.combin_update_dict,
            request_full_grad_params_name = self.request_full_grad_params_name
        )


    def unpack_client_model(self, packed_params):
        # self.logger.log(f"Unpacking client model {packed_params.keys()}")
        return self.server_compress_combin.uncompress(packed_params["combin_alpha"], self.model.state_dict())

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""

        self.logger.log(f"Round {self.current_epoch}")

        public_model_byte = calculate_data_size(self.public_model_params, set_sparse=self.set_sparse,set_layout=self.set_layout)

        for key, value in self.combin_update_dict.items():
            public_model_byte += calculate_data_size(value, set_sparse=self.set_sparse, set_layout=self.set_layout)

        clients_package = self.trainer.train()
        
        for client_id, package in clients_package.items():
            self.clients_comm_recv_bytes[client_id] += public_model_byte
            byte = calculate_data_size(clients_package[client_id]['model_params_diff']["combin_alpha"], 
                                set_sparse=self.set_sparse, 
                                set_layout=self.set_layout)
            self.clients_comm_send_bytes[client_id] += byte
            
            package['model_params_diff_bak'] = package['model_params_diff']
            package["model_params_diff"] = self.unpack_client_model(package["model_params_diff"])

                    # clients_model_param_dict: {layer_name: [clent1_grad, client2_grad, ...]} in clients_package
        
        # 更新压缩器的基
        def get_vector_by_list(vector_list: List[torch.Tensor]):
            return torch.stack([vector.flatten() for vector in vector_list], dim=0)
        
        clients_model_param_dict = {
            key: get_vector_by_list([package["model_params_diff_bak"]["combin_alpha"][key] for package in clients_package.values()]) 
            for key in self.request_full_grad_params_name
        }
        self.combin_update_dict = self.server_compress_combin.update_basis_by_vector(clients_model_param_dict)
        
        # 更新全局模型
        self.aggregate(clients_package)

        # 更新请求全局梯度的参数名
        print("Updating request full grad params name...")
        if self.fixed_adj_freq == 0:
            new_request_full_grad_params_name = []
            for key in self.server_compress_combin.setting_dict.keys():
                error_norms = [package["model_params_diff_bak"]["combin_error_norm"][key] for package in clients_package.values()]
                avg_error_norm = sum(error_norms) / len(error_norms)
                print(f"Layer {key} avg_error_norm: {avg_error_norm}")
                # tensor_error = torch.tensor(error_norms)
                if key not in self.error_dict:
                    delta_error = 0
                    self.error_dict[key] = avg_error_norm
                    self.total_error_dict[key] = avg_error_norm
                else:
                    delta_error = avg_error_norm - self.error_dict[key]
                    self.error_dict[key] = avg_error_norm
                    self.total_error_dict[key] += avg_error_norm
                z = self.kp * avg_error_norm + self.ki * self.total_error_dict[key] + self.kd * delta_error
                self.logger.log((f"Layer {key} z: {z}, z_thr: {self.Z_thr.get(key, None)}"))

                if key not in self.Z_thr:
                    # 更新基的第0轮，key不在Z_thr中，但是key在request_full_grad_params_name中，此时什么都不做，也不需要更新基
                    # 更新基的第1轮，key依然不在Z_thr中，但第一轮时request_full_grad_params_name被清空，此时可以更新阈值
                    if key not in self.request_full_grad_params_name:
                        self.Z_thr[key] = z * self.gamma
                elif z > self.Z_thr[key]:
                    self.logger.log(f"Need update basis of {key}")
                    new_request_full_grad_params_name.append(key)
                    self.total_error_dict[key] = 0
                    self.Z_thr.pop(key, None)
            self.request_full_grad_params_name = new_request_full_grad_params_name
        else:
            self.request_full_grad_params_name = list(self.server_compress_combin.setting_dict.keys()) \
                    if (self.current_epoch + 1) % self.fixed_adj_freq == 0 else []
        
        print("New request full grad params name: ", self.request_full_grad_params_name)
