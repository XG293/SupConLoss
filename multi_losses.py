"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class MultiSupConLoss(nn.Module):
    """ 
        Revised from Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(MultiSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        torch.set_printoptions(threshold=10000)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        print("features:", features.size()) # torch.Size([bs, 2, 128])

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            print("labels", labels.size(), labels)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            #print("mask", mask.size(), mask) # [bsz, bsz]
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1] # 2
        #print("contrast_count(should be 2):", contrast_count)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        #print("contrast_feature shape(should be [2*bsz, 128]:", contrast_feature.size())
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            #print("contrast_mode is all")
            anchor_feature = contrast_feature # [2*bsz, 128]
            anchor_count = contrast_count #2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        #print("anchor_dot_contrast shape(should be [32, 32]):", anchor_dot_contrast.size())
        ### for numerical stability ###
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() 
        #print("logits shape(should be [32, 32]):", logits.size())
        #print("trace of logits(should be 0):", torch.trace(logits))
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        #print("mask repeat:", mask) # [32, 32] or [2*bsz, 2*bsz]
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        #print("mask", mask)
        #print("logits_mask", logits_mask)
        mask = mask * logits_mask
        #print("mask", mask)
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) #[32, 32]
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) # [32,]
        
        # loss
        loss_inter = - (self.temperature / self.base_temperature) * mean_log_prob_pos # [32,]

        loss_inter = loss_inter.view(anchor_count, batch_size).mean()

        exp_logits_intra = torch.exp(logits) * mask
        print("check mask one's amount:", mask.sum(1))
        log_prob_intra = logits - torch.log(exp_logits_intra.sum(1, keepdim=True)) #[32, 32]
        intra_mask = torch.eye(batch_size).repeat(anchor_count, contrast_count).to(device) - torch.eye(batch_size * anchor_count).to(device)
        mean_log_prob_pos_intra = (intra_mask * log_prob_intra).sum(1) / intra_mask.sum(1) # [32,]
        #print("intra mask:", intra_mask)
        #print("check intra mask one's amount(must be 1 for each row):", intra_mask.sum(1))
        loss_intra = - (self.temperature / self.base_temperature) * mean_log_prob_pos_intra # [32,]

        loss_intra = loss_intra.view(anchor_count, batch_size).mean()
        loss = loss_inter + loss_intra
        return loss
