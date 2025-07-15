import torch
from torch import nn, Tensor
from torch.nn import BCEWithLogitsLoss, HuberLoss, L1Loss
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
        #self.soft_segment_mean_loss = SoftSegmentMean()

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

        focal = self.focal_loss(predictions, target)
        huber = self.huber_loss(num_predicted_events, num_true_events)
        consec = torch.mean(torch.sum(probabilities[:,1:] * probabilities[:,:-1], dim=1))
        #softsegment = self.soft_segment_mean_loss(signals, probabilities, target)

        weights = Tensor([self.alpha, self.beta, self.gamma]).to(predictions.device)
        losses  = torch.stack((focal, huber, consec))
        return (weights @ losses), focal, huber, consec #, softsegment
        #return self.alpha*bce_loss + self.beta*huber_loss + self.gamma*consecutive_loss + self.delta*soft_segment_loss, bce_loss, huber_loss, consecutive_loss, soft_segment_loss

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
        probabilities = torch.sigmoid(predictions)
        ce_loss = self.bce(predictions, targets)
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean": loss = loss.mean()
        elif self.reduction == "sum": loss = loss.sum()
        return loss

### TODO unused ? 

class SoftSegmentMean(nn.Module):
    def __init__(self):
        super().__init__()

    def find_mu(self, s, signal):
        return torch.cumsum(s * signal, dim=1) / torch.cumsum(s, dim=1)

    def forward(self, signal: Tensor, predictions: Tensor, target: Tensor) -> Tensor:
        signal = torch.squeeze(signal[:, 0, :])
        sp = torch.cumsum(predictions, axis=1) + 1e-7
        st = torch.cumsum(target, axis=1) + 1e-7

        mu_p = self.find_mu(sp, signal)
        mu_t = self.find_mu(st, signal)

        final_loss = torch.abs(mu_t - mu_p)
        final_loss = torch.sum(final_loss, dim=1)
        final_loss = torch.mean(final_loss)

        return final_loss

class NormalizedL1(nn.Module):
    def __init__(self, margin=0):
        super(NormalizedL1, self).__init__()
        self.l1 = L1Loss(reduction='none')
        #print(f'Huber loss margin set to: {margin}')
        self.margin = margin

    def forward(self, predicted_number, true_number):
        margin_corrected_predicted = predicted_number - self.margin
        return (self.l1(margin_corrected_predicted, true_number) / (true_number + 1)).mean()

class SoftBorderLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, signal, predictions, target):
        signal = torch.squeeze(signal[:, 0, :])

        bp = torch.cumsum(predictions * signal, dim=1) + 1e-7
        bt = torch.cumsum(target * signal, dim=1) + 1e-7

        final_loss = torch.abs(bt - bp)
        final_loss = torch.sum(final_loss, dim=1)
        final_loss = torch.mean(final_loss)

        return final_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth = 1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions).view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2.*intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        return 1 - dice

def custom_loss(bce_f, huber_f, predictions, labels, alpha=0.05):
    num_predicted_events = torch.sum(torch.where(torch.sigmoid(torch.squeeze(predictions)) > 0.5, torch.tensor(1), torch.tensor(0)), dim=1)
    num_true_events = torch.sum(labels, dim=1)

    return bce_f(predictions, labels) + alpha*huber_f(num_true_events.squeeze(), num_predicted_events.squeeze())