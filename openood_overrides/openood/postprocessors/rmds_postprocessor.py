from typing import Any

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


def vectorized_relative_mahalanobis(features, class_mean, precision,
                                    whole_mean, whole_precision):
    """Return all class-relative Mahalanobis scores without a class loop."""
    centered = features - whole_mean.view(1, -1)
    background_scores = -torch.sum(
        torch.matmul(centered, whole_precision) * centered, dim=1
    )

    feature_precision = torch.matmul(features, precision)
    feature_quadratic = torch.sum(feature_precision * features, dim=1,
                                  keepdim=True)
    mean_precision = torch.matmul(class_mean, precision)
    mean_quadratic = torch.sum(mean_precision * class_mean, dim=1).view(1, -1)
    class_scores = (
        -feature_quadratic
        + 2.0 * torch.matmul(feature_precision, class_mean.t())
        - mean_quadratic
    )
    return class_scores - background_scores.view(-1, 1)


def precision_from_scatter(scatter: torch.Tensor, count: int) -> torch.Tensor:
    """Match EmpiricalCovariance's biased covariance and Hermitian inverse."""
    covariance = (scatter / float(count)).cpu().numpy().astype(np.float32)
    precision = scipy.linalg.pinvh(covariance, check_finite=False)
    return torch.from_numpy(precision).float()


class RMDSPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False
        self._device_statistics = None
        self.implementation_note = (
            'OpenOOD RMDS with algebraically equivalent streaming covariance '
            'and vectorized GPU scoring'
        )

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # Estimate the same biased covariance as EmpiricalCovariance while
            # retaining only O(CD + D^2), rather than O(ND), state.
            print('\n Estimating mean and variance from training set '
                  '(streaming equivalent)...')
            class_counts = None
            class_means = None
            within_scatter = None
            correct = 0
            total_seen = 0
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    labels = batch['label'].to(data.device)
                    logits, features = net(data, return_feature=True)
                    features = features.detach().float()
                    if class_counts is None:
                        feature_dim = int(features.shape[1])
                        class_counts = torch.zeros(
                            self.num_classes, device=features.device
                        )
                        class_means = torch.zeros(
                            self.num_classes, feature_dim,
                            device=features.device
                        )
                        within_scatter = torch.zeros(
                            feature_dim, feature_dim, device=features.device
                        )

                    unique, inverse = torch.unique(
                        labels, sorted=True, return_inverse=True
                    )
                    group_count = torch.bincount(
                        inverse, minlength=len(unique)
                    ).to(features.dtype)
                    group_sum = torch.zeros(
                        len(unique), features.shape[1],
                        dtype=features.dtype, device=features.device
                    )
                    group_sum.index_add_(0, inverse, features)
                    group_mean = group_sum / group_count[:, None]

                    # Scatter inside the current batch's class groups.
                    residual = features - group_mean[inverse]
                    within_scatter.add_(residual.t().matmul(residual))

                    # Parallel-variance correction when each group is merged
                    # into its running class mean.
                    old_count = class_counts[unique]
                    old_mean = class_means[unique]
                    merged_count = old_count + group_count
                    delta = group_mean - old_mean
                    coefficient = old_count * group_count / merged_count
                    correction = delta * coefficient.clamp_min(0).sqrt()[:, None]
                    within_scatter.add_(correction.t().matmul(correction))
                    class_means[unique] = (
                        old_mean + delta * (group_count / merged_count)[:, None]
                    )
                    class_counts[unique] = merged_count

                    correct += int(logits.argmax(1).eq(labels).sum())
                    total_seen += int(labels.numel())

            if not total_seen:
                raise RuntimeError('RMDS setup received an empty ID train loader')
            missing = torch.where(class_counts.eq(0))[0].cpu().tolist()
            if missing:
                raise RuntimeError(f'RMDS has empty classes: {missing[:20]}')
            print(f' Train acc: {correct / total_seen:.2%}')

            self.class_mean = class_means.cpu()
            self.whole_mean = (
                (class_means * class_counts[:, None]).sum(0) / total_seen
            ).cpu()
            between = (
                (class_means - self.whole_mean.to(class_means.device))
                * class_counts.sqrt()[:, None]
            )
            whole_scatter = within_scatter + between.t().matmul(between)
            self.precision = precision_from_scatter(
                within_scatter, total_seen
            )
            self.whole_precision = precision_from_scatter(
                whole_scatter, total_seen
            )
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net(data, return_feature=True)
        pred = logits.argmax(1)
        device = features.device
        if (self._device_statistics is None
                or self._device_statistics[0].device != device):
            self._device_statistics = tuple(
                value.to(device)
                for value in (
                    self.class_mean, self.precision,
                    self.whole_mean, self.whole_precision,
                )
            )
        class_scores = vectorized_relative_mahalanobis(
            features, *self._device_statistics
        )
        conf = torch.max(class_scores, dim=1)[0]
        return pred, conf
