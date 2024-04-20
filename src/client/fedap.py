from typing import Any

import torch

from src.client.fedbn import FedBNClient


class FedAPClient(FedBNClient):
    def __init__(self, **commons):
        super(FedAPClient, self).__init__(**commons)

        self.model.need_all_features()
        self.pretrain = False

    def load_data_indices(self):
        super().load_data_indices()
        num_pretrain_samples = int(self.args.fedap.pretrain_ratio * len(self.trainset))
        if self.args.fedap.version != "f":
            if self.pretrain:
                self.trainset.indices = self.trainset.indices[:num_pretrain_samples]
            else:
                self.trainset.indices = self.trainset.indices[num_pretrain_samples:]

    def set_parameters(self, package: dict[str, Any]):
        self.pretrain = package["pretrain"]
        super().set_parameters(package)

    def package(self):
        client_package = super().package()
        if not self.pretrain:
            client_package["personal_model_params"].update(
                client_package["regular_model_params"]
            )
            del client_package["regular_model_params"]
        return client_package

    @torch.no_grad()
    def get_all_features(self, server_package: dict[str, Any]):
        self.set_parameters(server_package)
        features_list = []
        batch_size_list = []
        for x, _ in self.trainloader:
            features_list.append(
                [
                    feature.cpu()
                    for feature in self.model.get_all_features(x.to(self.device))
                ]
            )
            batch_size_list.append(len(x))

        if self.args.fedap.version == "d":
            for i, features in enumerate(features_list):
                for j in range(len(features)):
                    if len(features[j].shape) == 4 and len(features[j + 1].shape) < 4:
                        features_list[i] = [features[j]]
                        break
        return dict(features_list=features_list, batch_size_list=batch_size_list)
