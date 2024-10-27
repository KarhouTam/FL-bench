# 广播梯度压缩
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch

from src.server.fedavg import FedAvgServer
from src.client.fedtest4 import FedTest5Client
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


class FedTest4Server(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedTest4",
        unique_model=True,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.logger.log("Begin to initialize FedTest4Server.")
        self.init_trainer(FedTest5Client)
        print("Setting_dict: ", setting_dict)
        self.clients_compress_combin = {key: CompressorCombin(setting_dict, "SlideSVDCompressor") for key in range(self.client_num)}
        self.server_compress_combin = CompressorCombin(setting_dict, "SlideSVDCompressor")
        
        self.clients_last_global_model_params = {key: deepcopy(self.model.state_dict()) for key in range(self.client_num)}

        _ = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.combin_error = _
        # self.clients_combin_error = {key: deepcopy(_) for key in range(self.client_num)}
        self.total_weight = None

        self.aggregated_grad = {}
        # self.clients_combin_alpha = {key: {} for key in range(self.client_num)}
        # self.clients_combin_update_dict = {key: {} for key in range(self.client_num)}
        self.global_grad = {'combin_alpha':{}, 'combin_update_dict':{}}
        # self.clients_global_grad = {key: {'combin_alpha':{}, 'combin_update_dict':{}} for key in range(self.client_num)}
        self.error_params = {}
        # self.lf = LayerFilter(unselect_keys=list(setting_dict.keys()))


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
            last_global_model_params=self.clients_last_global_model_params[client_id],

            global_grad=self.global_grad,
            compress_combin=self.clients_compress_combin[client_id],

            # global_model = self.model,
            # combin_update_dict = self.clients_combin_update_dict[client_id],
            # combin_alpha = self.clients_combin_alpha[client_id],
        )

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        selected_clients = sorted(self.selected_clients)

        if self.global_grad['combin_alpha'] != {}:
            recv_byte = calculate_data_size(self.global_grad['combin_alpha'], set_sparse=self.set_sparse, set_layout=self.set_layout)
            for key, value in self.global_grad['combin_update_dict'].items():
                recv_byte += calculate_data_size(value, set_sparse=self.set_sparse, set_layout=self.set_layout)
        else:
            recv_byte = calculate_data_size(self.public_model_params, set_sparse=self.set_sparse, set_layout=self.set_layout)

        clients_package = self.trainer.train()
        for client_id in selected_clients:
            self.clients_personal_model_params[client_id] = clients_package[client_id]['personal_model_params']

        # for key in self.public_model_params.keys():
        #     if equal_tensor(last_global_model_params[][key], self.public_model_params[key]):
        #         print(f"Success: {key} === public model params.")
        #     else:
        #         print(f"Error: {key} is !== public model params.")
        #         print(f"Agg {key}: {aggregated[key]}\nPublic {key}: {self.public_model_params[key]}")
        # first_value = self.clients_last_global_model_params[selected_clients[0]]
        # 假设当前是第1轮本地训练结束，则first_value是第1轮的全局模型参数 == public_model_params(1)+e
        
        # client_grad = self.clients_compress_combin[selected_clients[0]].uncompress(self.global_grad['combin_alpha'], first_value)
        # for key in self.global_grad['combin_alpha'].keys():
        #     server_params = self.public_model_params[key].data + self.combin_error[key]
        #     if torch.allclose(first_value[key], server_params, atol=1e-6):
        #         print(f"Success: Client recover client_params {key} === recover server_params.")
        #     else:
        #         print(f"Error: Client recover client_params {key} !== recover server_params.")

        # print('-------------')

        self.aggregate(clients_package)

        if True:
        # if self.current_epoch % 5 != 0 or self.current_epoch == 0:
            # grad = {key: self.aggregated_grad[key] for key in self.aggregated_grad.keys()}
            combin_alpha, combin_update_dict, combin_error = self.server_compress_combin.compress(self.aggregated_grad, can_update_basis_func=lambda: True)
            self.combin_error = combin_error
            self.global_grad['combin_alpha'] = combin_alpha
            self.global_grad['combin_update_dict'] = combin_update_dict
        else:
            self.global_grad = {'combin_alpha':{}, 'combin_update_dict':{}}
            self.combin_error = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}

        # if self.current_epoch % 5 == 0 and self.current_epoch != 0:
        #     error_grad = self.server_compress_combin.uncompress(combin_alpha, self.model.state_dict())
        #     first_value = self.clients_last_global_model_params[selected_clients[0]]

        #     for key in self.public_model_params.keys():
        #         error_grad[key] = error_grad[key].to(first_value[key])
        #         new_global = first_value[key].data - error_grad[key]

        #         true_error_global = self.public_model_params[key].data + self.combin_error[key]
        #         if torch.allclose(new_global, true_error_global, atol=1e-6):
        #             print(f"Success: Client new_global {key} === the true error global param.")
        #         else:
        #             print(f"Error: Client new_global {key} !== the true error global param.")

        #     # 检测全部client的权重均值是否等于public_model_params
        # aggregated = self.__aggregate(clients_package, selected_clients)
        # for key in self.public_model_params.keys():
        #     if torch.allclose(aggregated[key], self.public_model_params[key], atol=1e-6):
        #         print(f"Success: {key} === public model params.")
        #     else:
        #         print(f"Error: {key} is !== public model params.")
        #         print(f"Agg {key}: {aggregated[key]}\nPublic {key}: {self.public_model_params[key]}")

        for client_id in selected_clients:
            self.clients_comm_recv_bytes[client_id] += recv_byte

            byte = calculate_data_size(clients_package[client_id]['model_params_diff'], 
                                    set_sparse=self.set_sparse, 
                                    set_layout=self.set_layout)
            self.clients_comm_send_bytes[client_id] += byte

            ############################################################
            # Update the combin_alpha
            ############################################################
            # combin_error = self.clients_combin_error[client_id]
            # grad = {key: self.aggregated_grad[key] - client_grad[key] for key in self.aggregated_grad.keys()}
            # self.clients_global_grad[client_id] = grad
            # self.clients_global_grad[client_id]['combin_alpha'] = combin_alpha
            # self.clients_global_grad[client_id]['combin_update_dict'] = combin_update_dict

    # def __aggregate(self, clients_package: OrderedDict[int, dict[str, Any]], selected_clients: list[int]):
    #     clients_weight = [package["weight"] for package in clients_package.values()]
    #     weights = torch.tensor(clients_weight) / sum(clients_weight)
    #     res = {}
    #     for name, global_param in self.public_model_params.items():
    #         diffs = torch.stack(
    #             [
    #                 self.clients_personal_model_params[client_id][name]
    #                 for client_id in self.clients_personal_model_params
    #             ],
    #             dim=-1,
    #         )
    #         res[name] = torch.sum(
    #             diffs * weights, dim=-1, dtype=global_param.dtype
    #         ).to(global_param.device)

    #     return res

    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):
        clients_weight = [package["weight"] for package in clients_package.values()]
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
            self.public_model_params[name].data += self.combin_error[name].to(self.public_model_params[name].device)
            error_norm = self.combin_error[name].norm()
            self.logger.log(f"Layer {name} error_norm: {error_norm}")