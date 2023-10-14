import torch
import torch.nn as nn
import torch.nn.functional as F

from transform import target_grid


class TrackNetLoss:
    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

    def __call__(self, preds, batch):
        # preds = [[10*50*80*80]]
        preds = preds[0].to(self.device) # only pick first (stride = 16)
        batch_target = batch['target'].to(self.device)

        loss = torch.zeros(2, device=self.device)  # box, cls, dfl
        batch_size = preds.shape[0]
        # for each batch
        for idx, pred in enumerate(preds):
            # pred = [50 * 80 * 80]
            pred_distri, pred_scores = torch.split(pred, [40, 10], dim=0)

            targets = pred_distri.clone().detach().to(self.device)
            cls_targets = torch.zeros(10, pred_scores.shape[1], pred_scores.shape[2], device=self.device)
            stride = self.stride[0]
            for idx, target in enumerate(batch_target[idx]):
                if target[1] == 1:
                    # xy
                    grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                    targets[4*idx, grid_y, grid_x] = offset_x
                    targets[4*idx + 1, grid_y, grid_x] = offset_y
                    targets[4*idx + 2, grid_y, grid_x] = target[4]
                    targets[4*idx + 3, grid_y, grid_x] = target[5]

                    ## cls
                    cls_targets[idx, grid_y, grid_x] = 1
            weight = 10
            loss[0] += weight * F.mse_loss(pred_distri, targets, reduction='mean')
            loss[1] += focal_loss(pred_scores, cls_targets, alpha=[0.94, 0.06], weight=weight)

        return loss.sum() * batch_size, loss.detach()

def focal_loss(pred_logits, targets, alpha=0.95, gamma=4.0, epsilon=1e-6, weight=10):
    """
    :param pred_logits: 預測的logits, shape [batch_size, 1, H, W]
    :param targets: 真實標籤, shape [batch_size, 1, H, W]
    :param alpha: 用於平衡正、負樣本的權重。這裡可以是一個scalar或一個list[alpha_neg, alpha_pos]。
    :param gamma: 用於調節著重於正確或錯誤預測的程度
    :return: focal loss
    """
    pred_probs = torch.sigmoid(pred_logits)
    # pred_probs = torch.clamp(pred_probs, epsilon, 1.0 - epsilon)
    if isinstance(alpha, (list, tuple)):
        alpha_neg = alpha[0]
        alpha_pos = alpha[1]
    else:
        alpha_neg = (1 - alpha)
        alpha_pos = alpha

    pt = torch.where(targets == 1, pred_probs, 1 - pred_probs)
    alpha_t = torch.where(targets == 1, alpha_pos, alpha_neg)
    
    ce_loss = -torch.log(pt)
    fl = alpha_t * (1 - pt) ** gamma * ce_loss
    fl = torch.where(targets == 1, fl * weight, fl)
    return fl.mean()