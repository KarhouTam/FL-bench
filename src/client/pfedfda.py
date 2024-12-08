from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.correlation_tools import cov_nearest
from torch.utils.data import DataLoader, Subset
from torchmin import minimize

from src.client.fedavg import FedAvgClient
from src.utils.constants import NUM_CLASSES


class pFedFDAClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.features_length = self.model.classifier.in_features
        num_clients = len(self.data_indices)
        self.label_distributions: List[torch.Tensor] = [None] * num_clients

        for client_id in range(num_clients):
            trainloader = DataLoader(
                Subset(self.dataset, indices=self.data_indices[client_id]["train"])
            )
            labels = []
            for _, y in trainloader:
                labels.append(y)

            labels = torch.cat(labels)
            label_counts = torch.bincount(
                labels, minlength=NUM_CLASSES[self.args.dataset.name]
            ).float()
            label_distribution = label_counts / len(labels)
            label_distribution += self.args.pfedfda.eps
            label_distribution /= label_distribution.sum()
            self.label_distributions[client_id] = label_distribution.to(self.device)

        self.global_means = torch.zeros(
            (NUM_CLASSES[self.args.dataset.name], self.features_length)
        )
        self.global_covariances = torch.zeros(self.features_length)
        self.local_means = self.global_means.clone()
        self.local_covariances = self.global_covariances.clone()
        self.adaptive_means = self.global_means.clone()
        self.adaptive_covariances = self.global_covariances.clone()
        self.means_beta = torch.ones(NUM_CLASSES[self.args.dataset.name])
        self.covariances_beta = torch.tensor(0.5)

    def package(self):
        package = super().package()
        package["local_means"] = self.local_means.cpu().clone()
        package["local_covariances"] = self.local_covariances.cpu().clone()
        package["adaptive_means"] = self.adaptive_means.cpu().clone()
        package["adaptive_covariances"] = self.adaptive_covariances.cpu().clone()
        package["means_beta"] = self.means_beta.cpu().clone()
        package["covariances_beta"] = self.covariances_beta.cpu().clone()
        return package

    def set_parameters(self, package):
        super().set_parameters(package)
        self.global_means = package["global_means"].to(self.device)
        self.global_covariances = package["global_covariances"].to(self.device)
        self.local_means = package["local_means"].to(self.device)
        self.local_covariances = package["local_covariances"].to(self.device)
        self.adaptive_means = package["adaptive_means"].to(self.device)
        self.adaptive_covariances = package["adaptive_covariances"].to(self.device)
        self.means_beta = package["means_beta"].to(self.device)
        self.covariances_beta = package["covariances_beta"].to(self.device)
        if self.testing:
            self.model.eval()
            features, labels = self.compute_features(self.testloader)
            self.compute_beta_values(features, labels)
            means_mle, scatter_mle, counts = self.compute_mle_statistics(
                features, labels
            )
            means_mle = torch.stack(
                [
                    (
                        means_mle[i]
                        if means_mle[i] is not None
                        and counts[i] > self.args.pfedfda.num_cv_folds
                        else self.global_means[i]
                    )
                    for i in range(NUM_CLASSES[self.args.dataset.name])
                ]
            )
            cov_mle = (
                (scatter_mle / (np.sum(counts) - 1))
                + self.args.pfedfda.eps
                + torch.eye(self.features_length, device=self.device)
            )
            cov_psd = torch.tensor(
                cov_nearest(cov_mle.cpu().numpy(), method="clipped"), device=self.device
            )
            self.local_means = means_mle
            self.local_covariances = cov_psd
            self.adaptive_means = (
                self.means_beta.unsqueeze(1) * means_mle
                + (1 - self.means_beta.unsqueeze(1)) * self.global_means
            )
            self.adaptive_covariances = (
                self.covariances_beta * cov_psd
                + (1 - self.covariances_beta) * self.global_covariances
            )
            self.set_lda_weights(
                means=self.adaptive_means, covariance=self.adaptive_covariances
            )

    def compute_beta_values(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Solves for beta values used in the model.

        This function computes the beta means and covariance based on the
        provided features and labels. If the local beta option is enabled,
        it initializes means and covariance to ones. Otherwise, it prunes
        features from classes with fewer than K samples and employs
        Stratified K-Fold cross-validation to compute the means and covariances.

        Args:
            features (torch.Tensor): Input features of shape (n_samples, n_features).
            labels (torch.Tensor): Corresponding labels of shape (n_samples,).
        """
        if self.args.pfedfda.local_beta:  # use only local
            self.means_beta = torch.ones(size=(1,), device=self.device)
            self.covariances_beta = torch.ones(size=(1,), device=self.device)
            return

        unique_labels, label_counts = np.unique(labels, return_counts=True)
        filtered_features = features.clone()
        filtered_labels = labels.clone()

        # Remove classes with < K samples, as we cannot do StratifiedKFold
        for label, count in zip(unique_labels, label_counts):
            if count < self.args.pfedfda.num_cv_folds:
                filtered_features = filtered_features[filtered_labels != label]
                filtered_labels = filtered_labels[filtered_labels != label]

        try:
            skf = StratifiedKFold(n_splits=self.args.pfedfda.num_cv_folds, shuffle=True)
            test_features_list, test_labels_list = [], []
            mean_list, cov_list = [], []

            for train_indices, test_indices in skf.split(
                filtered_features, filtered_labels
            ):
                train_features, train_labels = (
                    filtered_features[train_indices],
                    filtered_labels[train_indices],
                )
                test_features, test_labels = (
                    filtered_features[test_indices],
                    filtered_labels[test_indices],
                )

                means_mle, scatter_mle, train_label_counts = (
                    self.compute_mle_statistics(
                        features=train_features, labels=train_labels
                    )
                )

                train_cov_matrix = (
                    (scatter_mle / (np.sum(train_label_counts) - 1))
                    + 1e-4
                    + torch.eye(self.features_length, device=self.device)
                )
                cov_psd = torch.tensor(
                    cov_nearest(train_cov_matrix.cpu().numpy(), method="clipped"),
                    device=self.device,
                )

                means = torch.stack(
                    [
                        (
                            means_mle[i]
                            if means_mle[i] is not None
                            else self.global_means[i]
                        )
                        for i in range(NUM_CLASSES[self.args.dataset.name])
                    ]
                )

                mean_list.append(means)
                cov_list.append(cov_psd)
                test_features_list.append(test_features)
                test_labels_list.append(test_labels)

        except Exception:  # not enough data; use local stats
            self.means_beta = torch.zeros(size=(1,), device=self.device)
            self.covariances_beta = torch.zeros(size=(1,), device=self.device)
            return

        loss_function = lambda a: torch.sum(
            torch.stack(
                [
                    self.classify_features_with_lda(
                        a.clip(0, 1),
                        mean_list[i],
                        cov_list[i],
                        test_features_list[i],
                        test_labels_list[i],
                    )
                    for i in range(self.args.pfedfda.num_cv_folds)
                ]
            )
        )

        try:
            if self.args.pfedfda.single_beta:
                result = (
                    minimize(
                        loss_function,
                        x0=0.5 * torch.ones(size=(1,), device=self.device),
                        method="l-bfgs",
                        max_iter=10,
                        options={"gtol": 1e-3},
                    )
                    .x.cpu()
                    .clip(0, 1)
                )
                self.means_beta = torch.ones_like(self.means_beta) * result[0]
                self.covariances_beta = result[0]
            else:
                result = (
                    minimize(
                        loss_function,
                        x0=0.5 * torch.ones(size=(2,), device=self.device),
                        method="l-bfgs",
                        max_iter=10,
                        options={"gtol": 1e-3},
                    )
                    .x.cpu()
                    .clip(0, 1)
                )
                self.means_beta = torch.ones_like(self.means_beta) * result[0]
                self.covariances_beta = result[1]

        except Exception:  # if optimization fails, use last used value of beta
            pass

    def compute_mle_statistics(
        self,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor | None], torch.Tensor, torch.Tensor, np.ndarray]:
        """Compute Maximum Likelihood Estimation (MLE) statistics.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.
            features (Optional[torch.Tensor], optional): Input features. If None, they will be computed. Defaults to None.
            labels (Optional[torch.Tensor], optional): Input labels. If None, they will be computed. Defaults to None.

        Returns:
            Tuple[List[torch.Tensor | None], torch.Tensor, torch.Tensor, np.ndarray]:
                - List of mean features for each class.
                - Scatter matrix of features.
                - Tensor representation of the label distribution.
                - Count of labels for each class.
        """

        # Initialize means for each class
        num_classes = NUM_CLASSES[self.args.dataset.name]
        means = [None] * num_classes

        # Calculate label counts and distribution
        label_counts = np.bincount(labels.numpy(), minlength=num_classes)

        # Compute means for each class based on available features
        for y in torch.unique(labels).tolist():
            if label_counts[y] > self.args.pfedfda.num_cv_folds:
                means[y] = torch.mean(features[labels == y], dim=0)

        # Center features by class mean
        features_centered: List[torch.Tensor] = []
        for y in torch.unique(labels).tolist():
            means_y = means[y] if means[y] is not None else self.global_means[y]
            features_centered.append(features[labels == y] - means_y)

        features_centered = torch.cat(features_centered)

        # Calculate scatter matrix
        scatter = features_centered.T @ features_centered

        return means, scatter, label_counts

    @torch.no_grad()
    def compute_features(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes features and collect labels from the provided dataloader.

        Args:
            dataloader (DataLoader): A DataLoader containing the input data and labels.

        Returns:
            tuple: A tuple containing two tensors:
                - features (torch.Tensor): The extracted features from the model.
                - labels (torch.Tensor): The corresponding labels for the input data.
        """
        features: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []

        # Set the model to evaluation mode and move to the specified device
        self.model.eval()

        # Iterate over the data from the DataLoader
        for x, y in dataloader:
            # Get last features from the model
            feature = self.model.get_last_features(x.to(self.device), detach=True)
            features.append(feature)
            labels.append(y)

        # Concatenate features and labels
        features = torch.cat(features)
        labels = torch.cat(labels)

        return features, labels

    def classify_features_with_lda(
        self,
        beta: float | List[float],
        local_means: torch.Tensor,
        local_covariances: torch.Tensor,
        features: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Classifies features using a linear discriminant analysis (LDA)
        approach with adjustable beta for local and global statistics.

        Args:
            beta (float | List[float]): The weighting factor(s) for combining
                local and global means and covariances. Can be a single float or
                a list of floats.
            local_means (torch.Tensor): Local means calculated from the
                features.
            local_covariances (torch.Tensor): Local covariances calculated from
                the features.
            features (torch.Tensor): The input features to classify.
            true_labels (torch.Tensor): The true labels corresponding to the input
                features.

        Returns:
            torch.Tensor: The cross-entropy loss calculated between the
                predicted labels and the true labels.
        """
        # Calculate means and covariances based on the provided beta
        if self.args.pfedfda.single_beta:
            means = beta * local_means + (1.0 - beta) * self.global_means
            covariances = (
                beta * local_covariances + (1.0 - beta) * self.global_covariances
            )
        else:
            means = beta[0] * local_means + (1.0 - beta[0]) * self.global_means
            covariances = (
                beta[-1] * local_covariances
                + (1.0 - beta[-1]) * self.global_covariances
            )

        # Classify features using LDA
        y_pred = self.classify_with_lda(
            features, means=means, covariances=covariances, use_least_squares=True
        )

        # Calculate and return the cross-entropy loss
        return torch.nn.functional.cross_entropy(y_pred, true_labels)

    def classify_with_lda(
        self,
        features: torch.Tensor,
        means: torch.Tensor,
        covariances: torch.Tensor,
        use_least_squares: bool = True,
    ) -> torch.Tensor:
        """Performs Linear Discriminant Analysis (LDA) classification.

        This method classifies input data using LDA by computing the
        coefficients and intercepts based on the means and covariances
        of the classes. It also takes into account label distribution
        for class priors.

        Args:
            Z (torch.Tensor): Input data of shape (n_samples, n_features) to be classified.
            means (torch.Tensor): The class means of shape
                (n_classes, n_features). Defaults to None.
            covariance_matrix (torch.Tensor): The covariance matrix
                of shape (n_features, n_features). Defaults to None.
            use_lstsq (bool, optional): Flag to determine whether to use
                least squares or matrix inversion for computing coefficients.
                Defaults to True.

        Returns:
            torch.Tensor: The classification scores for each sample.
        """
        # Regularize the covariance matrix
        covariances = ((1 - self.args.pfedfda.eps) * covariances) + (
            self.args.pfedfda.eps
            * torch.trace(covariances)
            / self.features_length
            * torch.eye(self.features_length, device=self.device)
        )

        # Compute coefficients using least squares or matrix inversion
        if use_least_squares:
            coefficients = torch.linalg.lstsq(covariances, means.T)[0].T
        else:
            coefficients = (torch.linalg.inv(covariances) @ means.T).T
        # Calculate intercepts for the LDA decision function
        intercepts = -0.5 * torch.diag(torch.matmul(means, coefficients.T)) + torch.log(
            self.label_distributions[self.client_id]
        )

        # Compute and return the classification scores
        return features @ coefficients.T + intercepts

    def set_lda_weights(
        self, means: torch.Tensor, covariance: torch.Tensor, use_lstsq: bool = True
    ) -> None:
        """Set the LDA weights for the model classifier.

        This function computes the coefficients and intercepts for the linear discriminant analysis (LDA)
        based on the provided means and covariance. It optionally uses least squares to compute the coefficients.

        Args:
            means (torch.Tensor): A tensor representing the means of each class.
            covariance (torch.Tensor): A tensor representing the covariance matrix.
            use_lstsq (bool): A flag indicating whether to use least squares for coefficient calculation.
                            Defaults to True.

        Returns:
            None: This function modifies the classifier weights and biases in place.
        """
        with torch.no_grad():
            covariance = ((1 - self.args.pfedfda.eps) * covariance) + (
                self.args.pfedfda.eps
                * covariance.trace()
                / self.features_length
                * torch.eye(self.features_length, device=self.device)
            )

            if use_lstsq:
                coefs = torch.linalg.lstsq(covariance, means.T)[0].T
            else:
                coefs = torch.matmul(torch.linalg.inv(covariance), means.T).T

            intercepts = -0.5 * torch.diag(torch.matmul(means, coefs.T)) + torch.log(
                self.label_distributions[self.client_id]
            )

            self.model.classifier.weight.data = coefs
            self.model.classifier.bias.data = intercepts
