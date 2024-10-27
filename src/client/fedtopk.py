from copy import deepcopy
import torch
from src.utils.compressor_utils import TopkCompressor
from src.client.fedavg import FedAvgClient

class FedTopkClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.compressor = TopkCompressor(
            self.args.fedtopk.topk, 
            self.args.fedtopk.sparse_format)

    def pack_client_model(self, raw_model:dict[str, torch.Tensor] , global_model):
        packet_to_send = {}
        for key in raw_model.keys():
            packet_to_send[key], _ = self.compressor.compress(global_model[key] - raw_model[key])
        return packet_to_send

    def package(self):
        """Package data that client needs to transmit to the server.
        You can override this function and add more parameters.

        Returns:
            A dict: {
                `weight`: Client weight. Defaults to the size of client training set.
                `regular_model_params`: Client model parameters that will join parameter aggregation.
                `model_params_diff`: The parameter difference between the client trained and the global. `diff = global - trained`.
                `eval_results`: Client model evaluation results.
                `personal_model_params`: Client model parameters that absent to parameter aggregation.
                `optimzier_state`: Client optimizer's state dict.
                `lr_scheduler_state`: Client learning rate scheduler's state dict.
            }
        """
        model_params = self.model.state_dict()
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            regular_model_params={
                key: model_params[key].clone().cpu() for key in self.regular_params_name
            },
            personal_model_params={
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
        )
        if self.return_diff:
            client_package["model_params_diff"] = self.pack_client_model(
                client_package["regular_model_params"],
                self.global_regular_model_params,
            )
            client_package.pop("regular_model_params")
        return client_package