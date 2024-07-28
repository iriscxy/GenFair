import pdb

import torch
import torch.nn as nn


def RankingLoss(score, summary_score=None, margin=0, gold_margin=0, no_gold=True, no_cand=False):
    ones = torch.ones_like(score)
    loss_func = torch.nn.MarginRankingLoss(0.0, reduction='none')
    TotalLoss = loss_func(score, score, ones)
    TotalLoss = TotalLoss.mean(1)
    n = score.size(1)
    if not no_cand:
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='none')
            loss = loss_func(pos_score, neg_score, ones)
            loss = loss.view(-1, n - i)
            TotalLoss += loss.mean(dim=1)  # 在非批次维度上取平均
    if no_gold:
        return TotalLoss
    # gold summary loss
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin, reduction='none')
    loss = loss_func(pos_score, neg_score, ones)
    loss = loss.view(-1, n)  # 重新整形以匹配批次维度
    TotalLoss += loss.mean(dim=1)  # 在非批次维度上取平均
    return TotalLoss


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        batch_size=int(input.size()[0]/3)#16

        input_moredim=input.reshape(batch_size,3,input.size(1),input.size(2))
        target_moredim=target.reshape(batch_size,3,-1)
        mask_moredim=mask.reshape(batch_size,3,-1)
        input_first=input_moredim[:,0,:,:]#([16, 99, 3015]
        target_first=target_moredim[:,0,:]#[16, 99]
        mask_first=mask_moredim[:,0,:]#[16, 99]
        output = -input_first.gather(2, target_first.long().unsqueeze(2)).squeeze(2) * mask_first  # ([16, 99]
        mle_loss = torch.sum(output, dim=1) / torch.sum(mask_first, dim=1)

        input_second=input_moredim[:,1,:,:]
        target_second=target_moredim[:,1,:]
        mask_second=mask_moredim[:,1,:]
        output = input_second.gather(2, target_second.long().unsqueeze(2)).squeeze(2) * mask_second  # ([16, 99]
        second_score = torch.sum(output, dim=1) / torch.sum(mask_second, dim=1)

        input_third= input_moredim[:, 2, :, :]
        target_third = target_moredim[:, 2, :]
        mask_third = mask_moredim[:, 2, :]
        output = input_third.gather(2, target_third.long().unsqueeze(2)).squeeze(2) * mask_third  # ([16, 99]
        third_score = torch.sum(output, dim=1) / torch.sum(mask_third, dim=1)
        candidate_score = torch.stack([second_score, third_score], dim=1)
        rank_loss = RankingLoss(candidate_score, -mle_loss)

        combine_loss = mle_loss+rank_loss
        sorted_indices = torch.argsort(combine_loss, descending=True)
        high_loss_indices = sorted_indices[:int(len(sorted_indices) * 0.5)]
        final_loss = combine_loss[high_loss_indices]
        rank_loss_select=rank_loss[high_loss_indices]
        return final_loss.mean(),rank_loss_select.mean()

def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    loss ,rank_loss= criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])
    return loss,rank_loss