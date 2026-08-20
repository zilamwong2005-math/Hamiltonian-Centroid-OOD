from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class DICEPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(DICEPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.p = self.args.p
        self.mean_act = None
        self.masked_w = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False
        self.implementation_note = (
            'OpenOOD DICE with algebraically equivalent streaming feature mean'
        )

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            feature_sum = None
            sample_count = 0
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)
                    batch_sum = feature.detach().sum(dim=0).cpu().double()
                    feature_sum = (batch_sum if feature_sum is None else
                                   feature_sum + batch_sum)
                    sample_count += int(feature.shape[0])

            if not sample_count:
                raise RuntimeError('DICE setup received an empty ID train loader')
            self.mean_act = (feature_sum / sample_count).float().numpy()
            self.setup_flag = True
        else:
            pass

    def calculate_mask(self, w):
        contrib = self.mean_act[None, :] * w.data.squeeze().cpu().numpy()
        self.thresh = np.percentile(contrib, self.p)
        mask = torch.Tensor((contrib > self.thresh)).cuda()
        self.masked_w = w * mask

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        fc_weight, fc_bias = net.get_fc()
        if self.masked_w is None:
            self.calculate_mask(torch.from_numpy(fc_weight).cuda())
        _, feature = net(data, return_feature=True)
        vote = feature[:, None, :] * self.masked_w
        output = vote.sum(2) + torch.from_numpy(fc_bias).cuda()
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred, energyconf

    def set_hyperparam(self, hyperparam: list):
        self.p = hyperparam[0]

    def get_hyperparam(self):
        return self.p
