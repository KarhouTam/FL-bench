# 服务端相似阈值广播模型参数

from collections import OrderedDict
from copy import deepcopy

import torch
from src.server.fedavg import FedAvgServer
from src.client.fedtest1 import FedTest1Client
from src.utils.tools import NestedNamespace
from src.utils.my_utils import CKA, calculate_data_size, cos_similar
from src.utils.constants import OUT_DIR


class FedTest1Server(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedTest1",
        unique_model=True,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(FedTest1Client, output_dir=self.output_dir)
        
        self.set_sparse = []
        self.set_layout=None

        self.server_threshold = self.args.common.server_threshold
        self.similarity_func = self.args.common.server_similarity_fun

        self.latest_broadcast = None
        self.selected_name = []
        self.bak_log = self.logger.log
        self.logger.log = self.log

        if self.similarity_func == 'CKA':
            self.similarity_func = CKA
            self.logger.log("Use CKA similarity function")
        elif self.similarity_func == 'cosine':
            self.similarity_func = cos_similar
            self.logger.log("Use cosine similarity function")
        else:
            raise NotImplementedError(f"Similarity function {self.similarity_func} not implemented")
    
    def log(self, *args, **kwargs):
        # 加上时间戳 [年-月-日 时:分:秒]
        self.bak_log(f"Round {self.current_epoch}", *args, **kwargs)

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        selected_clients = sorted(self.selected_clients)
        self.latest_broadcast, self.selected_name = self.update_threshold_param(self.public_model_params, self.latest_broadcast)
        byte = calculate_data_size({key: self.latest_broadcast[key] for key in self.selected_name})

        clients_package = self.trainer.train()
        for client_id in selected_clients:
            self.clients_comm_recv_bytes[client_id] += byte
            self.clients_comm_send_bytes[client_id] += \
                calculate_data_size(clients_package[client_id]['model_params_diff'], 
                                    set_sparse=self.set_sparse, 
                                    set_layout=self.set_layout)

        self.aggregate(clients_package)


    def update_threshold_param(self, new_param: OrderedDict[str, torch.Tensor], last_param: OrderedDict[str, torch.Tensor]):
        updated_param = last_param
        selected_name = []
        if last_param is None:
            selected_name = new_param.keys()
            updated_param = deepcopy(new_param)
            self.logger.log(f"First broadcast param: {new_param.keys()}")
        else:
            for layername, layerparam in new_param.items():
                if layername in last_param:
                    sp = last_param[layername].shape
                    v = self.similarity_func(last_param[layername].reshape(-1, sp[-1]), layerparam.reshape(-1, sp[-1]))
                    if v >= self.server_threshold:
                        continue
                selected_name.append(layername)
                updated_param[layername] = layerparam.clone().detach()
        
        self.log(f"Update lower than threshold param: {selected_name} ")
        return updated_param, selected_name

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
        self.log(f"Client {client_id} personal_params: {personal_params.keys()}")
        return dict(
            regular_model_params=regular_params, 
            personal_model_params=personal_params,
        )