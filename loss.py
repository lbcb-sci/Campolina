import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.nn import BCEWithLogitsLoss, HuberLoss, L1Loss

class CustomLoss(_Loss):
    def __init__(self, alpha, beta, gamma, delta, focal_alpha, focal_gamma, eta, huber_delta, margin=0, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super().__init__(size_average, reduce, reduction)
        #self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        #self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        #self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.margin = margin
        #self.bce_loss = BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.huber_loss = HuberLoss(delta=huber_delta)
        self.normalizedl1 = NormalizedL1(margin=margin)
        self.soft_segment_mean_loss = SoftSegmentMean()

    def forward(self, signals, predictions, target):
        predicted_probabilities = torch.sigmoid(self.eta*predictions)
        num_predicted_events = torch.sum(predicted_probabilities, dim=1).float() - self.margin
        num_true_events = torch.sum(target, dim=1).float()
        #print(num_predicted_events, num_true_events)
        bce_loss = self.bce_loss(predictions, target)
        #huber_loss = self.normalizedl1(num_predicted_events, num_true_events)
        huber_loss = self.huber_loss(num_predicted_events, num_true_events)
        consecutive_loss = torch.mean(torch.sum(predicted_probabilities[:,1:] * predicted_probabilities[:,:-1], dim=1))
        soft_segment_loss = self.soft_segment_mean_loss(signals, predicted_probabilities, target)
        return self.alpha*bce_loss + self.beta*huber_loss + self.gamma*consecutive_loss + self.delta*soft_segment_loss, bce_loss, huber_loss, consecutive_loss, soft_segment_loss
        #return huber_loss, None, huber_loss


class FocalLoss(_Loss):
    def __init__(self, alpha, gamma, weight=None, size_average=True, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets):
        #inputs = inputs.float()
        #targets = targets.float()
        p = torch.sigmoid(predictions)
        ce_loss = self.bce(predictions, targets)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class SoftSegmentMean(nn.Module):
    def __init__(self):
        super().__init__()

    def find_mu(self, s, signal):
        return torch.cumsum(s * signal, dim=1) / torch.cumsum(s, dim=1)

    def forward(self, signal, predictions, target):
        signal = torch.squeeze(signal[:, 0, :])
        sp = torch.cumsum(predictions, axis=1) + 1e-7
        st = torch.cumsum(target, axis=1) + 1e-7

        mu_p = self.find_mu(sp, signal)
        mu_t = self.find_mu(st, signal)

        final_loss = torch.abs(mu_t - mu_p)
        final_loss = torch.sum(final_loss, dim=1)
        final_loss = torch.mean(final_loss)

        return final_loss


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


class NormalizedL1(nn.Module):
    def __init__(self, margin=0):
        super(NormalizedL1, self).__init__()
        self.l1 = L1Loss(reduction='none')
        print(f'Huber loss margin set to: {margin}')
        self.margin = margin

    def forward(self, predicted_number, true_number):
        margin_corrected_predicted = predicted_number - self.margin
        return (self.l1(margin_corrected_predicted, true_number) / (true_number + 1)).mean()



def custom_loss(bce_f, huber_f, predictions, labels, alpha=0.05):
    num_predicted_events = torch.sum(torch.where(torch.sigmoid(torch.squeeze(predictions)) > 0.5, torch.tensor(1), torch.tensor(0)), dim=1)
    num_true_events = torch.sum(labels, dim=1)

    return bce_f(predictions, labels) + alpha*huber_f(num_true_events.squeeze(), num_predicted_events.squeeze())