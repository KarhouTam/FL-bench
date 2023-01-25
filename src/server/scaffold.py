from copy import deepcopy
from typing import List, OrderedDict

import torch

from fedavg import FedAvgServer
from client.scaffold import SCAFFOLDClient
from config.args import get_scaffold_argparser
from config.utils import trainable_params


class SCAFFOLDServer(FedAvgServer):
    def __init__(self):
        super().__init__(
            "SCAFFOLD", get_scaffold_argparser().parse_args(), default_trainer=False
        )
        self.trainer = SCAFFOLDClient(deepcopy(self.model), self.args, self.logger)
        self.c_global = [
            torch.zeros_like(param) for param in trainable_params(self.model)
        ]

    def train(self):
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAININg EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]

            y_delta_cache = []
            c_delta_cache = []
            for client_id in self.selected_clients:

                client_local_params = self.generate_client_params(client_id)

                (
                    y_delta,
                    c_delta,
                    self.clients_metrics[client_id][E],
                ) = self.trainer.train(
                    client_id=client_id,
                    new_parameters=client_local_params,
                    c_global=self.c_global,
                    evaluate=self.args.eval,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )

                y_delta_cache.append(y_delta)
                c_delta_cache.append(c_delta)

            self.aggregate(y_delta_cache, c_delta_cache)

            self.log_info()

    @torch.no_grad()
    def aggregate(
        self,
        y_delta_cache: List[OrderedDict[str, torch.Tensor]],
        c_delta_cache: List[List[torch.Tensor]],
    ):
        y_delta_list = [list(delta.values()) for delta in y_delta_cache]
        for param, y_delta in zip(
            trainable_params(self.global_params_dict), zip(*y_delta_list)
        ):
            x_delta = torch.stack(y_delta).mean(dim=0).to(self.device)
            param.data += self.args.global_lr * x_delta

        # update global control
        for c_global, c_delta in zip(self.c_global, zip(*c_delta_cache)):
            c_delta = torch.stack(c_delta).mean(dim=0)
            c_global.data += (
                len(self.selected_clients) / self.client_num_in_total
            ) * c_delta


if __name__ == "__main__":
    server = SCAFFOLDServer()
    server.run()
