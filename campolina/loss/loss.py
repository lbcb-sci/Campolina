import torch
from torch import nn, Tensor
from torch.nn import BCEWithLogitsLoss, HuberLoss
from torch.nn.modules.loss import _Loss

class CustomLoss(_Loss):
    def __init__(
            self, 
            alpha: float, 
            beta: float, 
            gamma: float, 
            delta: float, 
            focal_alpha: float, 
            focal_gamma: float, 
            eta: float, 
            huber_delta: float, 
            margin: float = 0, 
            size_average = None, 
            reduce = None, 
            reduction = 'mean', 
        ):

        super().__init__(size_average, reduce, reduction)

        self.alpha = alpha; self.beta = beta; self.gamma = gamma
        self.delta = delta; self.eta = eta; self.margin = margin

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.huber_loss = HuberLoss(delta=huber_delta)

    @staticmethod
    def from_dict(scope: dict):
        return CustomLoss(
            alpha=scope['bce_alpha'], 
            beta=scope['huber_beta'], 
            gamma=scope['consecutive_gamma'], 
            delta=scope['softmean_delta'],
            focal_alpha=scope['focal_alpha'], 
            focal_gamma=scope['focal_gamma'], 
            eta=scope['logit_eta'],
            huber_delta=scope['huber_delta'],
            margin=scope['huber_margin'],
        )

    def forward(self, signals: Tensor, predictions: Tensor, target: Tensor) -> Tensor:
        probabilities = torch.sigmoid(self.eta * predictions)

        num_predicted_events = torch.sum(probabilities, dim=1).float() - self.margin
        num_true_events = torch.sum(target, dim=1).float()

        focal  = self.focal_loss(predictions, target)
        huber  = self.huber_loss(num_predicted_events, num_true_events)
        consec = torch.mean(torch.sum(probabilities[:,1:] * probabilities[:,:-1], dim=1))

        weights = Tensor([self.alpha, self.beta, self.gamma]).to(predictions.device)
        losses  = torch.stack((focal, huber, consec))

        return (weights @ losses), focal, huber, consec #, softsegment

class FocalLoss(_Loss):
    def __init__(
            self, 
            alpha: float, 
            gamma: float, 
            size_average = True, 
            reduce = None, 
            reduction = 'mean',
        ):
        super().__init__(size_average, reduce, reduction)

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        ce_loss = self.bce(predictions, targets)
        probabilities = torch.sigmoid(predictions)
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        loss = ce_loss * ((1 - p_t)**self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean": loss = loss.mean()
        elif self.reduction == "sum": loss = loss.sum()
        return loss
