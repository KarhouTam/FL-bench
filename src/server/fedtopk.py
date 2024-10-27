from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any

from src.server.fedavg import FedAvgServer
from src.client.fedtopk import FedTopkClient
from src.utils.tools import NestedNamespace
from src.utils.my_utils import calculate_data_size
from src.utils.compressor_utils import TopkCompressor



class FedTopkServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--topk", type=int, default=1)
        parser.add_argument("--sparse_format", type=str, default="csr")
        return parser.parse_args(args=args_list)
    
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "topk",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.init_trainer(FedTopkClient)
        self.compressor = TopkCompressor(args.fedtopk.topk, args.fedtopk.sparse_format)

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        selected_clients = sorted(self.selected_clients)
        
        public_model_byte = calculate_data_size(self.public_model_params, set_sparse=self.set_sparse,set_layout=self.set_layout)
        
        clients_package = self.trainer.train()

        for client_id in selected_clients:
            self.clients_comm_recv_bytes[client_id] += public_model_byte
            assert self.return_diff, "The return_diff must be True in top-k."
            try:
                byte = calculate_data_size(clients_package[client_id]['model_params_diff'], 
                                        set_sparse=self.set_sparse, 
                                        set_layout=self.set_layout)
            except:
                print(clients_package[client_id]['model_params_diff'])
            self.clients_comm_send_bytes[client_id] += byte

        self.aggregate(clients_package)

    def unpack_client_model(self, packed_model):
        # print(f'Unpacking model with {len(packed_model)} keys')
        for key in packed_model.keys():
            packed_model[key] = self.compressor.uncompress(packed_model[key], self.public_model_params[key].data.shape)
            # print(f'Uncompressed {key} with shape {packed_model[key].shape} vs global model shape {self.public_model_params[key].data.shape}')
        return packed_model


    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):
        for client_id, package in clients_package.items():
            package["model_params_diff"] = self.unpack_client_model(package["model_params_diff"])
        
        super().aggregate(clients_package)