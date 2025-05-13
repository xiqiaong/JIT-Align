import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss
import logging
from typing import Union

logger = logging.getLogger(__name__)

class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)
        
        # print("Target values:", target)
        # print("Prob shape:", prob.shape)
        # print("Prob size(1):", prob.size(1))

        assert (target >= 0).all() and (target < prob.size(1)).all(), "Index out of bounds"
        assert (prob >= 0).all() and (prob <= 1).all(), "Prob values should be between 0 and 1"

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        
        # 在 forward 函数中的合适位置添加以下代码段
        # assert (prob >= 0).all() and (prob <= 1).all(), "prob values should be between 0 and 1"

        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss

    
class BinaryFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        target = target.unsqueeze(dim=1)
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss
    
    
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 改为2,使得模型的输出的shape是[batchsize, 2]
        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.args = args

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config, args)
        self.args = args

    def forward(self, inputs_ids, attn_masks,
                labels=None):
        outputs = \
            self.encoder(input_ids=inputs_ids, attention_mask=attn_masks)[0]
        logits = self.classifier(outputs) # 用于分类

        # logger.info('logits:{}'.format(logits))
        # 激活函数，将logits张量中的每个元素的值压缩到[0,1]区间内
        prob = torch.sigmoid(logits)
        
        # 如果输入提供了labels标签
        if labels is not None:
            # loss_fct = BCELoss()
            # loss = loss_fct(prob, labels.unsqueeze(1).float())
            # return loss, prob
            loss_dp = BinaryFocalLoss(alpha=0.25, gamma=2, reduction='mean', num_class=2)
            loss = loss_dp(logits, labels)
            return loss, prob
        else:
            logger.info('labels is none!!!')
            return prob

