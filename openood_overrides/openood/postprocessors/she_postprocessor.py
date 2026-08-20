from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


def distance(penultimate, target, metric='inner_product'):
    if metric == 'inner_product':
        return torch.sum(torch.mul(penultimate, target), dim=1)
    elif metric == 'euclidean':
        return -torch.sqrt(torch.sum((penultimate - target)**2, dim=1))
    elif metric == 'cosine':
        return torch.cosine_similarity(penultimate, target, dim=1)
    else:
        raise ValueError('Unknown metric: {}'.format(metric))


class SHEPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(SHEPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.activation_log = None
        self.setup_flag = False
        self.implementation_note = (
            'OpenOOD SHE with algebraically equivalent streaming class means'
        )

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            net.eval()

            class_sums = None
            class_counts = None
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Eval: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    labels = batch['label'].to(data.device)

                    logits, features = net(data, return_feature=True)
                    if class_sums is None:
                        class_sums = torch.zeros(
                            self.num_classes,
                            features.shape[1],
                            dtype=features.dtype,
                            device=features.device,
                        )
                        class_counts = torch.zeros(
                            self.num_classes,
                            dtype=features.dtype,
                            device=features.device,
                        )
                    correct = logits.argmax(1).eq(labels)
                    if correct.any():
                        correct_labels = labels[correct]
                        class_sums.index_add_(
                            0, correct_labels, features[correct]
                        )
                        class_counts.index_add_(
                            0,
                            correct_labels,
                            torch.ones_like(
                                correct_labels, dtype=features.dtype
                            ),
                        )

            missing = torch.where(class_counts.eq(0))[0].cpu().tolist()
            if missing:
                raise RuntimeError(
                    'SHE found no correctly classified training sample for '
                    f'classes {missing[:20]}'
                )
            self.activation_log = (
                class_sums / class_counts[:, None]
            ).float()
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)
        pred = output.argmax(1)
        conf = distance(feature, self.activation_log[pred], self.args.metric)
        return pred, conf
