from argparse import Namespace
import torch

from fedavg import FedAvgServer
from src.config.args import get_fedavgm_argparser


class FedAvgMServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedAvgM",
        args: Namespace = None,
        unique_model=False,
        default_trainer=True,
    ):
        if args is None:
            args = get_fedavgm_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.global_optimizer = torch.optim.SGD(
            list(self.global_params_dict.values()),
            lr=1.0,
            momentum=self.args.server_momentum,
            nesterov=True,
        )

    @torch.no_grad()
    def aggregate(self, delta_cache, weight_cache):

        weights = torch.tensor(weight_cache, device=self.device) / sum(weight_cache)

        delta_list = [list(delta.values()) for delta in delta_cache]

        aggregated_delta = []
        for layer_delta in zip(*delta_list):
            aggregated_delta.append(
                torch.sum(
                    torch.stack(layer_delta, dim=-1).to(self.device) * weights, dim=-1
                )
            )

        self.global_optimizer.zero_grad()
        for param, diff in zip(self.global_params_dict.values(), aggregated_delta):
            param.grad = diff.data
        self.global_optimizer.step()


if __name__ == "__main__":
    server = FedAvgMServer()
    server.run()
