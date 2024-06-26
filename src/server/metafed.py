import time
from argparse import ArgumentParser, Namespace
from collections import OrderedDict

from rich.progress import track
from torch._tensor import Tensor

from src.client.metafed import MetaFedClient
from src.server.fedavg import FedAvgServer
from src.utils.tools import NestedNamespace


class MetaFedServer(FedAvgServer):

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--valset_ratio", type=float, default=0.2)
        parser.add_argument("--warmup_round", type=int, default=30)
        parser.add_argument("--lamda", type=float, default=1.0)
        parser.add_argument("--threshold_1", type=float, default=0.6)
        parser.add_argument("--threshold_2", type=float, default=0.5)
        return parser.parse_args(args_list)

    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "MetaFed",
        unique_model=True,
        use_fedavg_client_cls=False,
        return_diff=False,
    ):
        # NOTE: MetaFed does not support to select part of clients to train. So the join_raio is always set to 1.
        args.join_ratio = 1
        if args.mode == "parallel":
            print(
                "MetaFed does not support parallel training, so running mode is fallback to serial."
            )
            args.mode = "serial"
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
        self.warmup = True
        self.clients_flag = [True for _ in self.train_clients]
        self.init_trainer(MetaFedClient)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["local_epoch"] = (
            self.args.metafed.warmup_round
            if self.warmup
            else self.clients_local_epoch[client_id]
        )
        server_package["client_flag"] = self.clients_flag[client_id]
        return server_package

    def get_client_model_params(self, client_id: int) -> OrderedDict[str, Tensor]:
        params_dict = dict(
            student_model_params=self.clients_personal_model_params[client_id]
        )
        if not self.warmup:
            params_dict["teacher_model_params"] = OrderedDict(
                (
                    key,
                    self.clients_personal_model_params[
                        (client_id + self.client_num - 1) % self.client_num
                    ][key],
                )
                for key in self.public_model_param_names
            )
        return params_dict

    def train(self):
        # warm-up phase
        self.warmup = True
        # for self.update_client_params() works properly
        self.selected_clients = self.train_clients
        warmup_progress_bar = track(
            self.train_clients, "[bold cyan]Warming-up...", console=self.logger.stdout
        )

        for client_id in warmup_progress_bar:
            client_package = self.trainer.exec("warmup", [client_id])
            self.clients_personal_model_params[client_id].update(
                client_package[client_id]["client_model_params"]
            )
            self.clients_flag[client_id] = client_package[client_id]["client_flag"]

        self.warmup = False
        self.test()

        # client training phase
        avg_round_time = 0
        for E in self.train_progress_bar:
            self.current_epoch = E
            self.selected_clients = self.client_sample_stream[E]
            self.verbose = (E + 1) % self.args.common.verbose_gap == 0
            if self.verbose:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E + 1}", "-" * 26)

            begin = time.time()
            selected_clients_this_round = self.selected_clients
            for client_id in selected_clients_this_round:
                self.selected_clients = [client_id]
                client_package = self.trainer.train()
                self.clients_flag[client_id] = client_package[client_id]["client_flag"]
            end = time.time()
            self.log_info()
            avg_round_time = (avg_round_time * (self.current_epoch) + (end - begin)) / (
                self.current_epoch + 1
            )

            if (E + 1) % self.args.common.test_interval == 0:
                self.test()

        self.logger.log(
            f"{self.algo}'s average time taken by each global epoch: {int(avg_round_time // 60)} m {(avg_round_time % 60):.2f} s."
        )

        # personalization phase
        self.pers_progress_bar = track(
            self.train_clients,
            "[bold magenta]Personalizing...",
            console=self.logger.stdout,
        )
        self.current_epoch += 1
        for client_id in self.pers_progress_bar:
            client_package = self.trainer.exec("personalize", [client_id])
            self.clients_personal_model_params[client_id].update(
                client_package[client_id]["client_model_params"]
            )
            self.clients_metrics[client_id][self.current_epoch] = client_package[
                client_id
            ]["eval_results"]
        self.log_info()
        self.test()
