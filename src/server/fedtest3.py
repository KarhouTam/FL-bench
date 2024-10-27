# 广播域值+梯度压缩
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch

from src.server.fedavg import FedAvgServer
from src.client.fedtest3 import FedTest4Client
from src.utils.tools import NestedNamespace
from src.utils.my_utils import CKA, calculate_data_size, cos_similar
from src.utils.compressor_utils import CompressorCombin

# Debug日志
# 问题现象：一旦使用广播阈值，就会出现无法波动极大的情况，且很影响精度
# 1. 在fedtest1中，只使用了广播阈值，未发生问题
# 2. 在fedtest2中，只使用了上传压缩，未发生问题
# 3. 在fedtest3中，尝试使用大于1的阈值，即广播全部参数，未发生问题
# 4. 在fedtest3中, 尝试只使用广播阈值，将全部层都不参与上传压缩，表现和1一样
# 5. 是否是因为误差累积？尝试进行误差修正, 未发生大波动问题, 但是精度下降下降程度与未使用误差修正时相同
# 6. 是否是因为cosine相似度计算，后期基本不广播, 导致无法全局收敛？

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
    
    # 天选参数
    ############################################################################################################
    'classifier.fc1.weight' : (64, 32, 512), # shape: [4096, 9216]
    'classifier.fc1.bias' : (16, 8, 128),'classifier.bn6.weight' : (16, 8, 128),'classifier.bn6.bias' : (16, 8, 128),'classifier.bn6.running_mean' : (16, 8, 128),'classifier.bn6.running_var' : (16, 8, 128), # shape: [4096]

    'classifier.fc2.weight' : (64, 32, 512), # shape: [4096, 4096]
    'classifier.fc2.bias' : (2, 2, 64),'classifier.bn7.weight' : (2, 2, 64),'classifier.bn7.bias' : (2, 2, 64),'classifier.bn7.running_mean' : (2, 2, 64),'classifier.bn7.running_var' : (2, 2, 64), # shape: [4096]
    
    'classifier.fc3.weight' : (4, 4, 512),   # shape: [10, 4096]
    ############################################################################################################
    
    # # 'classifier.fc3.bias' : torch.Size([10]),
}

class FedTest3Server(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedTest3",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        
        self.init_trainer(FedTest4Client)
        print("Setting_dict: ", setting_dict)
        self.clients_compress_combin = {key: CompressorCombin(setting_dict, "SlideSVDCompressor") for key in range(self.client_num)}
        self.server_compress_combin = {key: CompressorCombin(setting_dict, "SlideSVDCompressor") for key in range(self.client_num)}
        _ = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.clients_combin_error = {key: deepcopy(_) for key in range(self.client_num)}

        self.set_sparse = []
        self.set_layout=None

        self.server_threshold = self.args.common.server_threshold
        self.server_norm_ratio_threshold = self.args.common.server_norm_ratio_threshold
        self.similarity_func = self.args.common.server_similarity_fun

        self.latest_broadcast = None
        self.selected_name = []
        self.norm_broadcast = {}

        if self.similarity_func == 'CKA':
            self.similarity_func = CKA
            self.logger.log("Use CKA similarity function")
        elif self.similarity_func == 'cosine':
            self.similarity_func = cos_similar
            self.logger.log("Use cosine similarity function")
        else:
            raise NotImplementedError(f"Similarity function {self.similarity_func} not implemented")

    def update_threshold_param(self, new_param: OrderedDict[str, torch.Tensor], updated_param: OrderedDict[str, torch.Tensor]):
        selected_name = []
        norm_broadcast = {}
        if updated_param is None:
            selected_name = new_param.keys()
            updated_param = deepcopy(new_param)
            self.logger.log(f"Round {self.current_epoch}: First broadcast param: {new_param.keys()}")
        else:
            for layername, layerparam in new_param.items():
                if layername in updated_param:
                    sp = updated_param[layername].shape

                    grad = layerparam - updated_param[layername]
                    # v = self.similarity_func(updated_param[layername].reshape(-1, sp[-1]), grad.reshape(-1, sp[-1]))
                    updated_norm = updated_param[layername].norm()                    
                    grad_norm = grad.norm()
                    ratio = grad_norm / updated_norm
                    if ratio <= self.server_norm_ratio_threshold:
                    # if abs(v) >= self.server_threshold or ratio <= self.server_norm_ratio_threshold:
                        norm_broadcast[layername] = layerparam.norm()
                        print(f"Layername: {layername}, weight norm: {updated_norm} grad norm: {grad_norm} ratio: {grad_norm/updated_norm}")
                        # print(f"Layername: {layername}, weight norm: {updated_norm} grad norm: {grad_norm} ratio: {grad_norm/updated_norm}, grad Similarity: {v} >= {self.server_threshold}")
                        continue
                    print(f"Layername: {layername}, weight norm: {updated_norm} grad norm: {grad_norm} ratio: {grad_norm/updated_norm}")
                    # print(f"Layername: {layername}, weight norm: {updated_norm} grad norm: {grad_norm} ratio: {grad_norm/updated_norm}, grad Similarity: {v}")
                selected_name.append(layername)
                updated_param[layername] = layerparam.clone().detach()
        
        self.logger.log(f"Round {self.current_epoch}: Update lower than threshold param: {selected_name} ")
        return updated_param, selected_name, norm_broadcast


    def get_client_model_params(self, client_id: int) -> OrderedDict[str, torch.Tensor]:
        """
        This function is for outputting model parameters that asked by `client_id`.

        Args:
            client_id (int): The ID of query client.

        Returns:
            {
                `regular_model_params`: Generally model parameters that join aggregation.
                `personal_model_params`: Client personal model parameters that won't join aggregation.
            }
        """
        regular_params = {key: self.latest_broadcast[key] for key in self.selected_name}
        personal_params = self.clients_personal_model_params[client_id]
        return dict(
            regular_model_params=regular_params, 
            personal_model_params=personal_params,
        )

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
            combin_error = self.clients_combin_error[client_id],
            norm_broadcast = self.norm_broadcast
        )

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        selected_clients = sorted(self.selected_clients)
        self.latest_broadcast, self.selected_name, self.norm_broadcast = self.update_threshold_param(self.public_model_params, self.latest_broadcast)
        recv_byte = calculate_data_size({key: self.latest_broadcast[key] for key in self.selected_name})
        recv_byte += calculate_data_size(self.norm_broadcast)

        clients_package = self.trainer.train()

        for client_id in selected_clients:
            self.clients_comm_recv_bytes[client_id] += recv_byte
            byte = calculate_data_size(clients_package[client_id]['model_params_diff']["combin_alpha"], 
                                    set_sparse=self.set_sparse, 
                                    set_layout=self.set_layout)
            for key, value in clients_package[client_id]['model_params_diff']["combin_update_dict"].items():
                byte += calculate_data_size(value, set_sparse=self.set_sparse, set_layout=self.set_layout)
            self.clients_comm_send_bytes[client_id] += byte

        self.aggregate(clients_package)

    def unpack_client_model(self, client_id, packed_params):
        templete_model_params = self.model.state_dict()
        compress_combin = self.server_compress_combin[client_id]
        compress_combin.update(packed_params["combin_update_dict"])
        uncompress = compress_combin.uncompress(packed_params["combin_alpha"], templete_model_params)
        return uncompress

    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):
        for client_id, package in clients_package.items():
            package["model_params_diff"] = self.unpack_client_model(client_id, package["model_params_diff"])
        
        super().aggregate(clients_package)
