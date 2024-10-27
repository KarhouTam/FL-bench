# 广播权重压缩 + 上传梯度压缩
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch

from src.server.fedavg import FedAvgServer
from src.client.fedtest6 import FedTest6Client
from src.utils.tools import NestedNamespace
from src.utils.my_utils import calculate_data_size
from src.utils.compressor_utils import CompressorCombin
from src.utils.constants import OUT_DIR


class FedTest6Server(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--upload_setting_dict", type=dict)
        parser.add_argument("--broadcast_setting_dict", type=dict)
        parser.add_argument("--u_type", type=str, default='float32')
        return parser.parse_args(args=args_list)

    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedTest6",
        unique_model=True,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.logger.log("Begin to initialize FedTest5Server.")
        self.init_trainer(FedTest6Client)
        if args.fedtest6.upload_setting_dict is None:
            upload_setting_dict = {}
        else:
            upload_setting_dict = args.fedtest6.upload_setting_dict.to_dict()
        if args.fedtest6.broadcast_setting_dict is None:
            broadcast_setting_dict = {}
        else:
            broadcast_setting_dict = args.fedtest6.broadcast_setting_dict.to_dict()
        u_dtype = args.fedtest6.u_type
        print("Upload Setting Dict: ", upload_setting_dict)
        print("Broadcast Setting Dict: ", broadcast_setting_dict)
        
        # 用于broadcast的
        self.clients_broadcast_compress_combin = {key: CompressorCombin(broadcast_setting_dict, "QuickSlideSVDCompressor", u_dtype=u_dtype) for key in range(self.client_num)}
        self.server_broadcast_compress_combin = CompressorCombin(broadcast_setting_dict, "QuickSlideSVDCompressor", u_dtype=u_dtype)
        _ = {key: torch.zeros_like(value, device=self.device) for key, value in self.model.named_parameters()}
        self.server_combin_error = _
        self.global_weight = {'combin_alpha':{}, 'combin_update_dict':{}}

        self.test_model = deepcopy(self.model)

        # 用于upload的
        self.clients_upload_compress_combin = {key: CompressorCombin(upload_setting_dict, "QuickSlideSVDCompressor", u_dtype=u_dtype) for key in range(self.client_num)}
        self.server_upload_compress_combin = {key: CompressorCombin(upload_setting_dict, "QuickSlideSVDCompressor", u_dtype=u_dtype) for key in range(self.client_num)}
        self.clients_combin_error = {key: deepcopy(_) for key in range(self.client_num)}
        self.total_weight = None

        # print("Device:",self.device)


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
            global_weight=self.global_weight,
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
        if self.global_weight['combin_alpha'] != {}:
            recv_byte = calculate_data_size(self.global_weight['combin_alpha'], set_sparse=self.set_sparse, set_layout=self.set_layout)
            for key, value in self.global_weight['combin_update_dict'].items():
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

            # 保存每轮的模型参数
            # save_model_param(clients_package[client_id]['personal_model_params'],
            #     self.current_epoch,
            #     f'c{client_id}',
            #     is_grad=True,
            #     path=OUT_DIR / self.algo / self.output_dir)

        self.aggregate(clients_package)

        # save_model_param(self.aggregated_grad,
        #     self.current_epoch,
        #     f'server',
        #     is_grad=True,
        #     path=OUT_DIR / self.algo / self.output_dir)
        # 用于broadcast的
        combin_alpha, combin_update_dict, combin_error = \
            self.server_broadcast_compress_combin.compress(self.public_model_params, can_update_basis_func=lambda: True)
        self.server_combin_error = combin_error
        self.global_weight['combin_alpha'] = combin_alpha
        self.global_weight['combin_update_dict'] = combin_update_dict


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
        weights = (torch.tensor(clients_weight) / sum(clients_weight)).to(self.device)
        for name, global_param in self.public_model_params.items():
            global_param.data = global_param.data.to(self.device)
            diffs = torch.stack(
                [
                    package["model_params_diff"][name]
                    for package in clients_package.values()
                ],
                dim=-1,
            )
            aggregated_grad = torch.sum(
                diffs * weights, dim=-1, dtype=global_param.dtype
            ).to(self.device)
            self.public_model_params[name].data -= self.server_combin_error[name]
            self.public_model_params[name].data -= aggregated_grad
            error_norm = self.server_combin_error[name].norm()
            self.logger.log(f"Layer {name} error_norm: {error_norm}")