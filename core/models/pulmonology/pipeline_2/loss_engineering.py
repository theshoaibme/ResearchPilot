import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# PIPELINE 2: MEDICAL LOSS FUNCTIONS (Focal Loss & Weighted CrossEntropy)
# ==============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in medical image classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss_criterion(loss_type: str = 'focal', class_weights=None, gamma: float = 2.0):
    """
    Returns the requested loss criterion.
    Options: 'focal', 'weighted_ce', 'cross_entropy'
    """
    if loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=gamma)
    elif loss_type == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
