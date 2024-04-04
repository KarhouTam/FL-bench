import time
from argparse import ArgumentParser, Namespace
from copy import deepcopy

from rich.progress import track

from fedavg import FedAvgServer
from src.client.metafed import MetaFedClient
from src.utils.tools import NestedNamespace


def get_metafed_args(args_list=None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--valset_ratio", type=float, default=0.2)
    parser.add_argument("--warmup_round", type=int, default=30)
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--threshold_1", type=float, default=0.6)
    parser.add_argument("--threshold_2", type=float, default=0.5)
    return parser.parse_args(args_list)


class MetaFedServer(FedAvgServer):
    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "MetaFed",
        unique_model=True,
        default_trainer=False,
    ):
        # NOTE: MetaFed does not support to select part of clients to train. So the join_raio is always set to 1.
        args.join_ratio = 1
        super().__init__(args, algo, unique_model, default_trainer)

        self.trainer = MetaFedClient(
            deepcopy(self.model), self.args, self.logger, self.device, self.client_num
        )
        self.warmup_progress_bar = track(self.train_clients, "[bold cyan]Warming-up...")
        self.pers_progress_bar = track(
            self.train_clients,
            "[bold magenta]Personalizing...",
            console=self.logger.stdout,
        )

    def train(self):
        # warm-up phase

        # for self.update_client_params() works properly
        self.selected_clients = self.train_clients

        client_params_cache = []
        self.trainer.local_epoch = self.args.metafed.warmup_round
        for client_id in self.warmup_progress_bar:
            client_local_params = self.generate_client_params(client_id)
            client_local_params = self.trainer.warmup(client_id, client_local_params)
            client_params_cache.append(client_local_params)
        self.update_client_params(client_params_cache)
        self.test()

        # client training phase
        avg_round_time = 0
        self.trainer.local_epoch = self.args.common.local_epoch
        for E in self.train_progress_bar:
            self.current_epoch = E
            begin = time.time()
            if (E + 1) % self.args.common.verbose_gap == 0:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E + 1}", "-" * 26)

            if (E + 1) % self.args.common.test_interval == 0:
                self.test()

            client_params_cache = []
            for client_id in self.selected_clients:
                student_params = self.generate_client_params(client_id)
                # teacher is the (i-1)-th client
                teacher_params = self.generate_client_params(
                    (client_id + self.client_num - 1) % self.client_num
                )
                student_params, self.client_metrics[client_id][E] = self.trainer.train(
                    client_id=client_id,
                    local_epoch=self.clients_local_epoch[client_id],
                    student_parameters=student_params,
                    teacher_parameters=teacher_params,
                    verbose=((E + 1) % self.args.common.verbose_gap) == 0,
                )
                self.trainer.update_flag()

                client_params_cache.append(student_params)

            self.update_client_params(client_params_cache)
            end = time.time()
            self.log_info()
            avg_round_time = (avg_round_time * (self.current_epoch) + (end - begin)) / (
                self.current_epoch + 1
            )

        self.logger.log(
            f"{self.algo}'s average time taken by each global epoch: {int(avg_round_time // 60)} m {(avg_round_time % 60):.2f} s."
        )

        # personalization phase
        self.current_epoch += 1
        client_params_cache = []
        for client_id in self.pers_progress_bar:
            student_params = self.generate_client_params(client_id)
            teacher_params = self.generate_client_params(
                (client_id + self.client_num - 1) % self.client_num
            )

            (student_params, self.client_metrics[client_id][self.current_epoch]) = (
                self.trainer.personalize(
                    client_id=client_id,
                    student_parameters=student_params,
                    teacher_parameters=teacher_params,
                )
            )

            client_params_cache.append(student_params)

        self.log_info()

        self.update_client_params(client_params_cache)
        self.test()
