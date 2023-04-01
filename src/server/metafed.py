from argparse import Namespace
from copy import deepcopy
from fedavg import FedAvgServer
from rich.progress import track
from tqdm import tqdm
from src.client.metafed import MetaFedClient
from src.config.args import get_metafed_argparser


class MetaFedServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "MetaFed",
        args: Namespace = None,
        unique_model=True,
        default_trainer=False,
    ):
        if args is None:
            args = get_metafed_argparser().parse_args()
        # NOTE: MetaFed does not support to select part of clients to train. So the join_raio is always set to 1.
        args.join_ratio = 1
        super().__init__(algo, args, unique_model, default_trainer)

        self.trainer = MetaFedClient(
            deepcopy(self.model), self.args, self.logger, self.client_num_in_total
        )
        self.warmup_progress_bar = (
            track(self.train_clients, "[bold cyan]Warming-up...", console=self.logger)
            if not self.args.save_log
            else tqdm(self.train_clients, "Warming-up..")
        )
        self.pers_progress_bar = (
            track(
                self.train_clients,
                "[bold magenta]Personalizing...",
                console=self.logger,
            )
            if not self.args.save_log
            else tqdm(self.train_clients, "Personalizing...")
        )

    def train(self):
        # warm-up phase

        # for self.update_client_params() works properly
        self.selected_clients = self.train_clients

        client_params_cache = []
        self.trainer.local_epoch = self.args.warmup_epoch
        for client_id in self.warmup_progress_bar:
            client_local_params = self.generate_client_params(client_id)
            client_local_params = self.trainer.warmup(client_id, client_local_params)
            client_params_cache.append(client_local_params)
        self.update_client_params(client_params_cache)
        self.test()

        # client training phase
        self.trainer.local_epoch = self.args.local_epoch
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E + 1}", "-" * 26)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            client_params_cache = []
            for client_id in self.selected_clients:
                student_params = self.generate_client_params(client_id)
                # teacher is the (i-1)-th client
                teacher_params = self.generate_client_params(
                    (client_id + self.client_num_in_total - 1)
                    % self.client_num_in_total
                )
                student_params, self.client_stats[client_id][E] = self.trainer.train(
                    client_id=client_id,
                    student_parameters=student_params,
                    teacher_parameters=teacher_params,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )
                self.trainer.update_flag()

                client_params_cache.append(student_params)

            self.update_client_params(client_params_cache)
            self.log_info()

        # personalization phase
        self.current_epoch += 1
        client_params_cache = []
        for client_id in self.pers_progress_bar:
            student_params = self.generate_client_params(client_id)
            teacher_params = self.generate_client_params(
                (client_id + self.client_num_in_total - 1) % self.client_num_in_total
            )

            (
                student_params,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.personalize(
                client_id=client_id,
                student_parameters=student_params,
                teacher_parameters=teacher_params,
            )

            client_params_cache.append(student_params)

        self.log_info()

        self.update_client_params(client_params_cache)
        self.test()


if __name__ == "__main__":
    server = MetaFedServer()
    server.run()
