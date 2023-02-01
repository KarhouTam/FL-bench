from copy import deepcopy
from fedavg import FedAvgServer
from config.args import get_fedrep_argparser
from client.fedrep import FedRepClient
from client.fedavg import FedAvgClient


class FedRepServer(FedAvgServer):
    def __init__(self):
        args = get_fedrep_argparser().parse_args()
        args.finetune_epoch = max(1, args.finetune_epoch)
        super().__init__("FedRep", args, default_trainer=False)
        self.trainer = FedRepClient(deepcopy(self.model), self.args, self.logger)

    def train(self):
        fedavg_trainer = FedAvgClient(deepcopy(self.model), self.args, self.logger)
        flag = True
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]

            delta_cache = []
            weight_cache = []
            for client_id in self.selected_clients:

                client_local_params = self.generate_client_params(client_id)
                if E < (self.args.global_epoch * 0.5):
                    trainer = fedavg_trainer
                else:
                    if flag:
                        flag = False
                        self.trainer.personal_params_dict = (
                            fedavg_trainer.personal_params_dict
                        )
                    trainer = self.trainer
                delta, weight, self.clients_metrics[client_id][E] = trainer.train(
                    client_id=client_id,
                    new_parameters=client_local_params,
                    evaluate=self.args.eval,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )

                delta_cache.append(delta)
                weight_cache.append(weight)

            self.aggregate(delta_cache, weight_cache)
            self.log_info()


if __name__ == "__main__":
    server = FedRepServer()
    server.run()
