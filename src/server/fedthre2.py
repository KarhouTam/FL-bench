from collections import OrderedDict
from copy import deepcopy
import time
from typing import Any

import torch
from src.server.fedavg import FedAvgServer
from src.client.fedthre2 import FedThre2Client
from src.utils.tools import NestedNamespace
from src.utils.my_utils import CKA, LayerFilter, calculate_data_size, cos_similar, save_model_param
from src.utils.constants import OUT_DIR
from src.utils.metrics import Metrics


class FedThre2Server(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedThre2",
        unique_model=True,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(FedThre2Client, output_dir=self.output_dir)
        # self.agg_lf = LayerFilter()
        self.agg_lf = LayerFilter(unselect_keys=['bn', 'running', 'num_batches_tracked'])
        self.clients_local_round_idx = [0] * self.client_num

        self.clients_total_opt_diff = [{} for i in range(self.client_num)]
        self.clients_grad_stack = [{} for i in range(self.client_num)]
        self.clients_grad_stack_recv = [{} for i in range(self.client_num)]
        
        # self.clients_comm_recv_bytes = [0] * self.client_num
        # self.clients_comm_send_bytes = [0] * self.client_num
        self.set_sparse = ['fc2.weight', 'fc3.weight']
        self.set_layout='torch.sparse_csr'

        self.total_weight = 1

        self.save_each_round = self.args.common.save_each_round
        self.server_threshold = self.args.common.server_threshold
        self.client_threshold = self.args.common.client_threshold
        self.grad_stack_len = self.args.common.grad_stack_len
        self.similarity_func = self.args.common.server_similarity_fun

        self.last_broadcast = None
        self.new_broadcast = {}
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
        self.update_threshold_param()
        selected_clients = sorted(self.selected_clients)
        for client_id in selected_clients:
            server_package = self.package(client_id)
            byte = calculate_data_size(server_package['regular_model_params'])
            self.clients_comm_recv_bytes[client_id] += byte
            # 输出server_package接收的personal_model_params.keys()和regular_model_params.keys()
            # self.logger.log(f"Client {client_id} received regular_model_params {byte}: {server_package['regular_model_params'].keys()} | personal_model_params: {server_package['personal_model_params'].keys()}")

        clients_package = self.trainer.train()
        for client_id in selected_clients:
            self.clients_local_round_idx[client_id] = clients_package[client_id]['local_round_idx']
            byte = calculate_data_size(clients_package[client_id]['model_params_diff'], 
                                       set_sparse=self.set_sparse, 
                                       set_layout=self.set_layout)
            self.clients_comm_send_bytes[client_id] += byte
            # 输出clients_package发送的personal_model_params.keys()和regular_model_params.keys()
            # self.logger.log(f"Client {client_id} sent regular_model_params {byte}: {clients_package[client_id]['regular_model_params'].keys()} | personal_model_params: {clients_package[client_id]['personal_model_params'].keys()}")
            if self.save_each_round:
                save_model_param(clients_package[client_id]['personal_model_params'],
                                self.clients_local_round_idx[client_id],
                                f'c{client_id}',
                                is_grad=False,
                                path=OUT_DIR / self.algo / self.output_dir)

        self.aggregate(clients_package)

    def update_threshold_param(self):
        new_param = self.public_model_params
        last_param = self.last_broadcast
        if last_param is None:
            self.new_broadcast = deepcopy(new_param)
            self.last_broadcast = deepcopy(new_param)
            self.logger.log(f"First broadcast param: {new_param.keys()}")
        else:
            res = {}
            for layername in self.agg_lf(last_param).keys():
                sp = last_param[layername].shape
                v = self.similarity_func(last_param[layername].reshape(-1, sp[-1]), new_param[layername].reshape(-1, sp[-1]))
                if v < self.server_threshold:
                    last_param[layername] = deepcopy(new_param[layername])
                    res[layername] = new_param[layername]
                self.logger.log(f"Layer {layername} similarity: {v}")
            self.new_broadcast = res
            self.logger.log(f"Update lower than threshold param: {res.keys()} ")

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
        if len(self.clients_personal_model_params[client_id].keys()) == 0:
            return dict(
                regular_model_params=self.new_broadcast,
                personal_model_params=self.clients_personal_model_params[client_id],
            )
        
        # set_regular_filter = LayerFilter(unselect_keys=['bn'])
        regular_params = deepcopy(self.agg_lf(self.new_broadcast))
        # self.logger.log(f"Client {client_id} received regular_model_params: {regular_params.keys()}")
        
        set_personal_filter = LayerFilter(unselect_keys=list(regular_params.keys()))
        personal_params = set_personal_filter(self.clients_personal_model_params[client_id])
        # self.logger.log(f"Client {client_id} received personal_model_params: {personal_params.keys()}")
        _ = set(self.public_model_params.keys()) - (set(regular_params.keys()).union(set(personal_params.keys())))
        assert _ == set(), f"Client model params miss {_}"
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
            return_diff=True,
            threshold=self.client_threshold,
            local_epoch=self.clients_local_epoch[client_id],
            **self.get_client_model_params(client_id),
            optimizer_state=self.clients_optimizer_state[client_id],
            lr_scheduler_state=self.clients_lr_scheduler_state[client_id],
            local_round_idx=self.clients_local_round_idx[client_id],
            total_opt_diff=self.clients_total_opt_diff[client_id],
            grad_stack=self.clients_grad_stack[client_id],
            grad_stack_len=self.grad_stack_len,
            total_weight=self.total_weight
        )
    

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
        self.total_weight = sum(clients_weight)
        # if self.return_diff:  # inputs are model params diff
        for name, global_param in self.agg_lf(self.public_model_params).items():
            diff_list = []
            norm_list = []
            for idx, package in clients_package.items():
                client_layer_diff = package["model_params_diff"][name]
                client_layer_diff['param'] = client_layer_diff['param'].to(self.device)
                if client_layer_diff['new_diff']:
                    norm_list.append(torch.norm(client_layer_diff['param'], p='fro'))
                    diff_list.append(client_layer_diff['param']/norm_list[-1])
                    if name not in self.clients_grad_stack_recv[idx]:
                        self.clients_grad_stack_recv[idx][name] = [client_layer_diff['param']]
                    else:
                        self.clients_grad_stack_recv[idx][name].append(client_layer_diff['param'])
                        if len(self.clients_grad_stack_recv[idx][name]) > self.grad_stack_len:
                            self.clients_grad_stack_recv[idx][name].pop(0)
                    self.logger.log(f"Client {idx} layer {name} update new diff")
                else:
                    G_params = self.clients_grad_stack_recv[idx][name]
                    sp = global_param.shape
                    if len(sp) != 1:
                        G = torch.stack(
                            [p.reshape(sp[0],-1) for p in G_params]
                            ).permute(1, 2, 0)
                    else:
                        G = torch.stack(
                            [p for p in G_params]
                            ).permute(1, 0).unsqueeze(0)
                    x = client_layer_diff['param'].to(self.device)
                    try:
                        Gx = torch.bmm(G, x.unsqueeze(-1)).squeeze(-1).reshape(sp)
                    except Exception as e:
                        self.logger.log(f"Client {idx} layer {name} G shape:{G.shape} x shape:{x.shape} error {e}")
                        save_model_param(G, self.current_epoch, f'c{idx}_error', pre_desc=f'G_{name}',is_grad=True)
                        save_model_param(x, self.current_epoch, f'c{idx}_error', pre_desc=f'x_{name}',is_grad=True)
                        exit(1)
                    norm_list.append(torch.norm(Gx, p='fro'))
                    diff_list.append(Gx.reshape(global_param.shape)/norm_list[-1])
                    self.logger.log(f"Client {idx} layer {name} use x shape:{client_layer_diff['param'].shape}")
                

            diffs = torch.stack(
                diff_list,
                dim=-1,
            )
            weights = (torch.tensor(clients_weight) / self.total_weight)
            aggregated = torch.sum(
                diffs.to(self.device) * weights.to(self.device), dim=-1, dtype=global_param.dtype
            )
            # 获取norm_list的中位数
            median_norm = torch.median(torch.stack(norm_list)).to(self.device)
            # 将aggregated长度变为mean_norm
            aggregated = aggregated * median_norm / torch.norm(aggregated, p='fro')
            self.public_model_params[name].data -= aggregated.to(global_param.device)