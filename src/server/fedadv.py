from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List

import torch
from src.server.fedavg import FedAvgServer
from src.client.fedadv import FedAdvClient
from src.utils.tools import NestedNamespace
from src.utils.my_utils import LayerFilter, aggregate_layer, calculate_data_size, get_config_for_round, save_model_param
from src.utils.constants import OUT_DIR


class FedAdvServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedAdv",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        super().__init__(args, f'{algo}-{args.common.desc}', unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(FedAdvClient)
        self.clients_local_round_idx = [0] * self.client_num
        self.clients_comm_recv_bytes = [0] * self.client_num
        self.clients_comm_send_bytes = [0] * self.client_num
        self.agg_cond:List[Dict[str, str]] = self.args.common.agg_cond_list
        self.save_each_round = self.args.common.save_each_round

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
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
            byte = calculate_data_size(clients_package[client_id]['regular_model_params'])
            self.clients_comm_send_bytes[client_id] += byte
            # 输出clients_package发送的personal_model_params.keys()和regular_model_params.keys()
            # self.logger.log(f"Client {client_id} sent regular_model_params {byte}: {clients_package[client_id]['regular_model_params'].keys()} | personal_model_params: {clients_package[client_id]['personal_model_params'].keys()}")
            if self.save_each_round:
                save_model_param({**clients_package[client_id]['regular_model_params'], 
                                  **clients_package[client_id]['personal_model_params']},
                                self.clients_local_round_idx[client_id],
                                f'c{client_id}',
                                is_grad=False,
                                path=OUT_DIR / self.algo / self.output_dir)

        self.aggregate(clients_package)

    def log_info(self):
        super().log_info()
        if self.args.common.visible == "visdom":
            raise NotImplementedError("Visdom is not supported in FedAdv.")
        elif self.args.common.visible == "tensorboard":
            self.tensorboard.add_scalar(
                f"Communicate/Client Total Receive Bytes",
                sum(self.clients_comm_recv_bytes),
                self.current_epoch,
                new_style=True,
            )
            self.tensorboard.add_scalar(
                f"Communicate/Client Total Send Bytes",
                sum(self.clients_comm_send_bytes),
                self.current_epoch,
                new_style=True,
            )


    def get_layer_filter(self, client_idx:int, usage:str):
        usage = usage.lower()
        if usage == 'post':
            round_idx = self.clients_local_round_idx[client_idx] + 1
        elif usage == 'set':
            round_idx = self.clients_local_round_idx[client_idx]
        if round_idx == 0:
            return LayerFilter()
        
        config = get_config_for_round(self.agg_cond, round_idx)

        assert config is not None, f"Client {client_idx} No condition satisfied"

        layer_filter = LayerFilter(
            unselect_keys = config.get('unselect_layer',None),
            all_select_keys = config.get('all_select_layer',None),
            any_select_keys = config.get('any_select_layer',None)
        )

        return layer_filter

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
        set_regular_filter = self.get_layer_filter(client_id, 'set')
        regular_params = deepcopy(set_regular_filter(self.public_model_params))
        
        set_personal_filter = LayerFilter(unselect_keys=list(regular_params.keys()))
        personal_params = set_personal_filter(self.clients_personal_model_params[client_id])
        _ = set(self.public_model_params.keys()) - (set(regular_params.keys()).union(set(personal_params.keys())))
        assert _ == set(), f"Client model params miss {_}"
        return dict(
            regular_model_params=regular_params, personal_model_params=personal_params
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
            local_round_idx=self.clients_local_round_idx[client_id],
            post_layer_filter=self.get_layer_filter(client_id, 'post')
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
        weights = torch.tensor(clients_weight) / sum(clients_weight)
        if self.return_diff:  # inputs are model params diff
            for name, global_param in self.public_model_params.items():
                diffs = torch.stack(
                    [
                        package["model_params_diff"][name]
                        for package in clients_package.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(
                    diffs * weights, dim=-1, dtype=global_param.dtype
                ).to(global_param.device)
                self.public_model_params[name].data -= aggregated
        else:
            for name, global_param in self.public_model_params.items():
                client_params = [
                    (package["weight"], {**package["regular_model_params"], **package["personal_model_params"]})
                    for package in clients_package.values()
                ]
                # aggregated = torch.sum(
                #     client_params * weights, dim=-1, dtype=global_param.dtype
                # ).to(global_param.device)
                aggregated = aggregate_layer(client_params, name, strict=False)
                if aggregated is not None:
                    global_param.data = aggregated

