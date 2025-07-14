####

import argparse

from pod5.tools.pod5_filter import parse_read_id_targets
from triton.language import dtype

import wandb
from tqdm import tqdm
import time

import os
import torch
import numpy as np
import pod5 as p5
import json

from torch import nn
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss, HuberLoss, L1Loss
from torch.nn.modules.loss import _Loss

from data.bam_util import BamIndex
from data.pod5_util import get_reads, get_reads_from_pod5, process_chunk
from model.model import EventDetector
from model.tcn_model import TCNEventDetector

#torch.manual_seed(12345)

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


def load_batches(bam_idx, pod5_path, batch_size, predict=False):   #TODO what if i preload a batch of batch_size signals and then yield batches as long as i can and then load batch_size signal again?
    current_batch = []
    current_borders = []
    current_identifiers = []

    for read in get_reads(pod5_path):
        for alignment in bam_idx.get_alignment(str(read.read_id)):
            if alignment is None:
                continue
            signal_chunks, chunk_borders, chunk_identifiers = process_chunk(aln=alignment, read=read, predict=predict, adjust_type=None)
            #print(f'Loaded {signal_chunks[0]}')
            #print(f'Loaded chunks {np.array(signal_chunks).shape} and borders {np.array(chunk_borders).shape}')
            #print(signal_chunks)
            #print(str(read.read_id))
            if signal_chunks is None:
                tqdm.write(f'Could not extract info for read {read.read_id}')
                continue
            #tqdm.write(f'Signal {read.read_id} has {len(signal_chunks)}')

            if len(current_batch) + len(signal_chunks) > batch_size:
                to_take = batch_size - len(current_batch)
                current_batch.extend(signal_chunks[:to_take])
                current_borders.extend(chunk_borders[:to_take])
                if predict:
                    current_identifiers.extend(chunk_identifiers[:to_take])
                    yield np.array(current_batch), np.array(current_borders), np.array(current_identifiers)
                else:
                    #print(read.read_id)
                    yield np.array(current_batch), np.array(current_borders)

                remaining = len(signal_chunks) - to_take
                while remaining >= batch_size:
                    current_batch = signal_chunks[to_take:to_take+batch_size]
                    current_borders = chunk_borders[to_take:to_take+batch_size]
                    if predict:
                        current_identifiers = chunk_identifiers[to_take:to_take+batch_size]
                        yield np.array(current_batch), np.array(current_borders), np.array(current_identifiers)
                    else:
                        #print(read.read_id)
                        yield np.array(current_batch), np.array(current_borders)

                    to_take = to_take + batch_size
                    remaining = remaining - batch_size

                current_batch = signal_chunks[to_take:]
                current_borders = chunk_borders[to_take:]
                if predict:
                    current_identifiers = chunk_identifiers[to_take:]

            else:
                current_batch.extend(signal_chunks)
                current_borders.extend(chunk_borders)
                if predict:
                    current_identifiers.extend(chunk_identifiers)
    if predict:
        yield np.array(current_batch), np.array(current_borders), np.array(current_identifiers)
    else:
        #print(read.read_id)
        yield np.array(current_batch), np.array(current_borders)


def test_model(bam_idx, model, device, loss_f, scope, valid=False):
    model.eval()
    full_predictions = []
    full_labels = []
    full_loss = 0.0
    full_bce_loss = 0.0
    full_huber_loss = 0.0
    full_consecutive_loss = 0.0
    full_softmean_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch, labels in tqdm(load_batches(bam_idx, scope['validation_pod5'], scope['val_batch_size'])):
            #batch = torch.unsqueeze(torch.Tensor(batch).to(device), 1)
            batch = torch.Tensor(batch).to(device)
            labels = torch.Tensor(labels).to(device)

            predictions = torch.squeeze(model(batch), dim=2)
            #print('Test prediction')
            #print(predictions)

            loss, bce_loss, huber_loss, consecutive_loss, softmean_loss = loss_f(batch, predictions, labels)
            #loss = loss_f(predictions, labels)
            full_loss += loss.item()
            full_bce_loss += bce_loss.item()
            full_huber_loss += huber_loss.item()
            full_consecutive_loss += consecutive_loss.item()
            full_softmean_loss += softmean_loss.item()
            steps += 1

            predicted_probabilities = torch.sigmoid(predictions)
            num_predicted_events = torch.sum(predicted_probabilities, dim=1).float()
            num_actual_events = torch.sum(labels, dim=1)
            #tqdm.write(f'Median difference between predicted and actual events: {torch.median(torch.abs(num_predicted_events  - num_actual_events))}')

            #tqdm.write(f'Validation predictions: {torch.sigmoid(torch.squeeze(predictions))[:,:20]}')
            #tqdm.write(f'Validation labels: {labels[:,:20]}')

            if not valid:
                predictions = np.where((1/(1 + np.exp(-predictions.detach().cpu().numpy()))) > 0.5, 1, 0)
                full_predictions.extend(list(predictions))
                full_labels.extend(labels.cpu().numpy())


    avg_loss = full_loss / steps
    avg_bce_loss = full_bce_loss / steps
    avg_huber_loss = full_huber_loss / steps
    avg_consecutive_loss = full_consecutive_loss / steps
    avg_softmean_loss = full_softmean_loss / steps
    return avg_loss, avg_bce_loss, avg_huber_loss, avg_consecutive_loss, avg_softmean_loss, full_predictions
    #return avg_loss


def train_step(batch, labels, model, device, loss_f, optimizer, total_steps):
    model.train()
    #batch = torch.unsqueeze(torch.Tensor(batch).to(device), 1)  #TODO when do I normalize the signal
    batch = torch.Tensor(batch).to(device)
    labels = torch.Tensor(labels).to(device)

    forward_start = time.time()
    predictions = torch.squeeze(model(batch), dim=2)
    #print('Train prediction')
    #print(predictions)
    forward_end = time.time()
    #if total_steps % 100 == 0:
    #    print(f'Forward took {forward_end - forward_start}')
    #print(predictions, labels)
    loss_start = time.time()
    loss, bce_loss, huber_loss, consecutive_loss, softmean_loss = loss_f(batch, predictions, labels)
    loss_end = time.time()
    #if total_steps % 100 == 0:
    #    print(f'Loss calculation took {loss_end - loss_start}')
    #loss = loss_f(predictions, torch.squeeze(labels))
    between_start = time.time()
    if total_steps % 3000 == 0:
        num_predicted_events = torch.sum(torch.where(torch.sigmoid(torch.squeeze(predictions)) > 0.5, torch.tensor(1), torch.tensor(0)), dim=1).int()
        num_true_events = torch.sum(labels, dim=1).int()
        #tqdm.write(f'Current BCE loss: {loss}')
        tqdm.write(f'Num predicted vs true num events:\n\t{num_predicted_events[:10]}\n\t{num_true_events[:10]}, '
                   f'alpha = {loss_f.alpha}, beta = {loss_f.beta}, gamma = {loss_f.gamma}')
    #    tqdm.write(f'BCE_loss: {bce_loss}, Huber loss: {huber_loss}, consecutive loss: {consecutive_loss}')
    if torch.isnan(loss):
        print(batch)
        print(labels)
        print(torch.sum(labels, dim=1))
        print(predictions)
    back_start = time.time()
    #if total_steps % 100 == 0:
    #    print(f'Between steps in train step took {back_start - between_start}')
    optimizer.zero_grad()
    loss.backward()
    back_end = time.time()
    #if total_steps % 100 == 0:
    #    print(f'Backward took {back_end - back_start}')
    step_start = time.time()
    optimizer.step()
    step_end = time.time()
    #if total_steps % 100 == 0:
    #    print(f'Optimizer step took {step_end - step_start}')

    return loss, bce_loss, huber_loss, consecutive_loss, softmean_loss, predictions


def train_epoch(bam_idx, model, device, optimizer, loss_f, scope, best_validation_loss, i, new_loss_step):
    model.train()

    total_loss = 0.0
    total_bce_loss = 0.0
    total_huber_loss = 0.0
    total_consecutive_loss = 0.0
    total_softmean_loss = 0.0
    total_examples = 0
    total_steps = 0

    patience = 0

    #validation_loss, validation_bce_loss, validation_huber_loss, validation_consecutive_loss, predictions = test_model(
    #    bam_idx=bam_idx, model=model, device=device, loss_f=loss_f,
    #    scope=scope, valid=True)
    load_start = time.time()
    for batch, borders in tqdm(load_batches(bam_idx=bam_idx, pod5_path=scope['train_pod5'], batch_size=scope['batch_size'], predict=False)):
        total_steps += 1
        loss, bce_loss, huber_loss, consecutive_loss, softmean_loss, train_predictions = train_step(batch=batch, labels=borders, model=model, device=device, optimizer=optimizer,
                   loss_f=loss_f, total_steps=total_steps)

        total_examples += len(batch)
        total_loss += loss.detach()
        total_bce_loss += bce_loss.detach()
        total_huber_loss += huber_loss.detach()
        total_consecutive_loss += consecutive_loss.detach()
        total_softmean_loss += softmean_loss.detach()


        if total_steps % 3000 == 0:
            validation_loss, validation_bce_loss, validation_huber_loss, validation_consecutive_loss, validation_softmean_loss, _ = test_model(bam_idx=bam_idx, model=model, device=device, loss_f=loss_f,
                                                      scope=scope, valid=True)
            #validation_loss = test_model(bam_idx=bam_idx, model=model, device=device, loss_f=loss_f, scope=scope, valid=True)
            tqdm.write(f'Validation loss after {total_steps} is {validation_loss}, BCE is {validation_bce_loss}, Huber is {validation_huber_loss}, Soft mean is {validation_softmean_loss}')
            #tqdm.write(f'Validation loss after {total_steps} is {validation_loss}')
            if best_validation_loss is None:
                best_validation_loss = validation_loss
                tqdm.write('Saving first version of the model')
                torch.save(model.state_dict(), scope['save_model'])

            if validation_loss < best_validation_loss:
                tqdm.write(f'Saving new version of model with the lowest validation loss: {validation_loss} < {best_validation_loss}')
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), scope['save_model'])

            else:
                patience += 1

            wandb.log(
                    {"epoch": i, "step": total_steps, "avg_train_loss": total_loss / total_steps, "avg_train_bce_loss": total_bce_loss / total_steps, "avg_train_huber_loss": total_huber_loss / total_steps, "avg_train_consecutive_loss": total_consecutive_loss / total_steps, "avg_train_softmean_loss": total_softmean_loss / total_steps,  "val_loss": validation_loss, "val_bce_loss": validation_bce_loss, "val_huber_loss": validation_huber_loss, "val_consecutive_loss": validation_consecutive_loss, "val_softmean_loss": validation_softmean_loss})
            #wandb.log({"epoch": i, "step": total_steps, "avg_train_loss": total_loss / total_steps, "val_loss": validation_loss})
        #if total_steps % 100 == 0:
        #    print(f'Post processing took {post_end - post_start}')
        load_start = time.time()
    total_loss = total_loss.item()
    avg_loss = total_loss / total_steps

    validation_loss, validation_bce_loss, validation_huber_loss, validation_consecutive_loss, validation_softmean_loss, predictions = test_model(bam_idx=bam_idx, model=model, device=device, loss_f=loss_f,
                                                      scope=scope, valid=True)
    tqdm.write(f'Validation loss after epoch {i} is {validation_loss}, BCE is {validation_bce_loss}, Huber is {validation_huber_loss}, consecutive {validation_consecutive_loss}, soft mean {validation_softmean_loss}')
    tqdm.write(
        f'Train loss after epoch {i} is {total_loss / total_steps}, BCE is {total_bce_loss / total_steps}, Huber is {total_huber_loss / total_steps}, consecutive {total_consecutive_loss / total_steps}, soft mean {total_softmean_loss / total_steps}')
    if i >= new_loss_step:
        if best_validation_loss is None:
                    best_validation_loss = validation_loss
                    tqdm.write('Saving first version of the model')
                    torch.save(model.state_dict(), scope['save_model'])
        if validation_loss < best_validation_loss:
                    tqdm.write(f'Saving new version of model with the lowest validation loss: {validation_loss} < {best_validation_loss}')
                    best_validation_loss = validation_loss
                    torch.save(model.state_dict(), scope['save_model'])

    if i % 5 == 0:
        borders = torch.Tensor(borders).detach().cpu()
        predictions = torch.Tensor(train_predictions).detach().cpu()
        print(f'Epoch: {i}')
        print(f'Predicted: {torch.sum(torch.where(predictions > 0, 1, 0), dim=1)}')
        print(f'Labels: {torch.sum(borders, dim=1)}')
        print(f'Correct positions {torch.sum(torch.where(predictions > 0, 1, 0)*borders, dim=1)}')
        print(f'True logits: {torch.sum(predictions*borders, dim=1)}')
        print(f'False logits: {torch.sum(predictions*(1-borders), dim=1)}')

    wandb.log(
        {"epoch": i, "step": total_steps, "avg_train_loss": total_loss / total_steps,
         "avg_train_bce_loss": total_bce_loss / total_steps, "avg_train_huber_loss": total_huber_loss / total_steps,
         "avg_train_consecutive_loss": total_consecutive_loss / total_steps,
         "avg_train_softmean_loss": total_softmean_loss / total_steps, "val_loss": validation_loss,
         "val_bce_loss": validation_bce_loss, "val_huber_loss": validation_huber_loss,
         "val_consecutive_loss": validation_consecutive_loss, "val_softmean_loss": validation_softmean_loss})

    return avg_loss, best_validation_loss


def main(scope):
    device = scope['devices'][0]
    wandb.init(project="event_detecting", entity="bakicsara97")

    bam_idx = BamIndex(scope['bam_file'])
    model = EventDetector(in_channels=scope['in_channels'], out_channels=scope['out_channels'], classification_head=scope['classification_head'], kernel_size_one=scope['kernel_one'], kernel_size_all=scope['kernel_all'])
    #model = TCNEventDetector(in_channels = scope['in_channels'], channels=scope['out_channels'], kernel_size=scope['kernel_all'], classification_head=scope['classification_head'], dropout=0.1, causal=False, use_norm='batch_norm', activation='gelu')
    model.to(device)
    wandb.watch(model)

    loss_f = CustomLoss(alpha=scope['bce_alpha'], beta=scope['huber_beta'], gamma=scope['consecutive_gamma'], delta=scope['softmean_delta'],
                        focal_alpha=scope['focal_alpha'], focal_gamma=scope['focal_gamma'], eta=scope['logit_eta'],
                        huber_delta=scope['huber_delta'], pos_weight=torch.Tensor([1]).to(device), margin=scope['huber_margin'])
    optimizer = AdamW(model.parameters(), lr=scope['lr'], eps=scope['adam_epsilon'])
    #pos_weight = torch.Tensor([10]).to(device)
    pos_weight = torch.Tensor([10]).to(device)
    #loss_f = BCEWithLogitsLoss(pos_weight=pos_weight)
    #loss_f = FocalLoss(alpha=0.8, gamma=2)
    #loss_f = DiceLoss()

    best_validation_loss = None

    for i in range(scope['epochs']):
        tqdm.write(f'Starting epoch {i}')
        #if i == scope['introduce_losses']:
            #loss_f.alpha = 1
        #    loss_f.beta = 5e-2
        #    loss_f.gamma = 1e-1
        train_loss, best_validation_loss = train_epoch(bam_idx=bam_idx, model=model, device=device, optimizer=optimizer, loss_f=loss_f, scope=scope, best_validation_loss=best_validation_loss, i=i, new_loss_step=scope['introduce_losses'])      #TODO myb do not evaluate on padded part of the signal

    print(f'Model saved to {scope["save_model"]}')
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='train_config.json')

    args = parser.parse_args()
    with open(args.config_file, 'r') as inf:
        scope = json.load(inf)

    if 'gpu' in scope and len(scope['gpu']) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((str(x) for x in scope['gpu']))
        scope["devices"] = [torch.device("cuda", x) for x in range(len(scope['gpu']))]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        scope["devices"] = [torch.device("cpu")]
    
    print(f'Using {scope["devices"]}')
    main(scope)
