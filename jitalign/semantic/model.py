import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss
import logging
from typing import Union
from torch.nn.functional import one_hot
from transformers import T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration, T5Config

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Computes the focal loss between input and target
    as described here https://arxiv.org/abs/1708.02002v2

    Args:
        gamma (float):  The focal loss focusing parameter.
        weights (Union[None, Tensor]): Rescaling weight given to each class.
        If given, has to be a Tensor of size C. optional.
        reduction (str): Specifies the reduction to apply to the output.
        it should be one of the following 'none', 'mean', or 'sum'.
        default 'mean'.
        ignore_index (int): Specifies a target value that is ignored and
        does not contribute to the input gradient. optional.
        eps (float): smoothing to prevent log from returning inf.
    """

    def __init__(
            self,
            gamma,
            weights: Union[None, Tensor] = None,
            reduction: str = 'mean',
            ignore_index=-100,
            eps=1e-16
    ) -> None:
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
            )
        assert weights is None or isinstance(weights, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(
                type(weights))
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.weights = weights

    def _get_weights(self, target: Tensor) -> Tensor:
        if self.weights is None:
            return torch.ones(target.shape[0])
        weights = target * self.weights
        return weights.sum(dim=-1)

    def _process_target(
            self, target: Tensor, num_classes: int, mask: Tensor
    ) -> Tensor:

        # convert all ignore_index elements to zero to avoid error in one_hot
        # note - the choice of value 0 is arbitrary, but it should not matter as these elements will be ignored in the loss calculation
        target = target * (target != self.ignore_index)
        target = target.view(-1)
        return one_hot(target, num_classes=num_classes)

    def _process_preds(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = torch.vstack([1 - x, x])
            x = x.permute(1, 0)
            return x
        return x.view(-1, x.shape[-1])

    def _calc_pt(
            self, target: Tensor, x: Tensor, mask: Tensor
    ) -> Tensor:
        p = target * x
        p = p.sum(dim=-1)
        p = p * ~mask
        return p

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError(
            'The predictions values should be between 0 and 1, \
                make sure to pass the values to sigmoid for binary \
                classification or softmax for multi-class classification'
        )
        mask = target == self.ignore_index
        mask = mask.view(-1)
        x = self._process_preds(x)
        num_classes = x.shape[-1]
        target = self._process_target(target, num_classes, mask)
        weights = self._get_weights(target).to(x.device)
        pt = self._calc_pt(target, x, mask)
        focal = 1 - pt
        nll = -torch.log(self.eps + pt)
        nll = nll.masked_fill(mask, 0)
        loss = weights * (focal ** self.gamma) * nll
        return self._reduce(loss, mask, weights)

    def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.sum() / (~mask * weights).sum()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x
        
        
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
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # AttributeError: 'T5Config' object has no attribute 'hidden_dropout_prob'
        self.dropout = nn.Dropout(0.1)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(config.hidden_size, 1)
        self.out_proj = nn.Linear(config.d_model, 1)
        self.args = args

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# T5ForConditionalGeneration
class Model(nn.Module):
    # def __init__(self, model, config, args):
    #     super(Model, self).__init__(config)
    #     # self.cls_head = nn.Linear(self.config.d_model, 2, bias=True)
    #     # self.encoder = model
    #     self.classifier = RobertaClassificationHead(config, args)
    #     # self.init()
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config, args)
        self.args = args

    def forward(self, inputs_ids, attn_masks,
                labels=None):
        
        # print("inputs_ids shape:", inputs_ids.shape)
        # print("attn_masks shape:", attn_masks.shape)
        # print("inputs_ids:", inputs_ids)
        # print("attn_masks:", attn_masks)

        outputs = self.encoder(input_ids=inputs_ids, attention_mask=attn_masks, output_attentions=False,
    return_dict=False)[0]
        
        # print("outputs:", outputs.shape)
        # print("outputs[:, 0, :]:", outputs[:, 0, :].shape)
        
        
        logits = self.classifier(outputs) # 用于分类

        # logger.info('logits:{}'.format(logits))
        prob = torch.sigmoid(logits)
        
        # CCT5-focalloss
        # weights = torch.FloatTensor([1, 3]).to(logits.device)
        # loss_fct = FocalLoss(gamma=3, weights=weights)
        # m = torch.nn.Softmax(dim=-1)
        
        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(prob, labels.unsqueeze(1).float())
            # return loss, prob
            # loss_dp = BinaryFocalLoss(alpha=0.25, gamma=2, reduction='mean', num_class=2)
            # loss = loss_dp(logits, labels)
            # loss = loss_fct(m(logits), labels)
            return loss, prob
        else:
            logger.info('labels is none!!!')
            return prob

