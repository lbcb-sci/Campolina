import logging
import torch.utils.benchmark
import wandb
import numpy as np

import torch
from torch.optim import AdamW

from campolina.data import BamIndex, load_batches
from campolina.model.model import EventDetector
from campolina.loss import CustomLoss

def train(scope: dict):
    device = scope['devices'][0]

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    wandb.init()

    logger.info('initializing model...')
    model = EventDetector(
        in_channels=scope['in_channels'], 
        out_channels=scope['out_channels'], 
        classification_head=scope['classification_head'], 
        kernel_size_one=scope['kernel_one'], 
        kernel_size_all=scope['kernel_all'],
    ).to(device)

    wandb.watch(model)

    logger.info('initializing loss function...')
    loss_f = CustomLoss(
        alpha=scope['bce_alpha'], 
        beta=scope['huber_beta'], 
        gamma=scope['consecutive_gamma'], 
        delta=scope['softmean_delta'],
        focal_alpha=scope['focal_alpha'], 
        focal_gamma=scope['focal_gamma'], 
        eta=scope['logit_eta'],
        huber_delta=scope['huber_delta'], 
        pos_weight=torch.Tensor([1]).to(device), 
        margin=scope['huber_margin'],
    )

    logger.info('initializing optimizer...')
    optimizer = AdamW(
        model.parameters(), 
        lr=scope['lr'], 
        eps=scope['adam_epsilon'],
    )

    best_val_loss = None

    bam_idx = BamIndex(scope['bam_file'])

    for epoch in range(scope['epochs']):

        print(f'\n-- Starting epoch {epoch}...')

        train_loss, best_val_loss = train_epoch(
            bam_idx=bam_idx, 
            model=model, 
            device=device, 
            optimizer=optimizer, 
            loss_f=loss_f, 
            scope=scope, 
            best_val_loss=best_val_loss, 
            epoch=epoch, 
            new_loss_step=scope['introduce_losses']
        ) # TODO myb do not evaluate on padded part of the signal

    print(f'Model saved to {scope["save_model"]}')
    wandb.finish()

def train_epoch(
        bam_idx: BamIndex, 
        model: EventDetector, 
        device: torch.device, 
        optimizer: torch.optim.Optimizer, 
        loss_f: CustomLoss, 
        scope: dict, 
        best_val_loss, 
        epoch: int, 
        new_loss_step,
    ):
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = total_bce_loss = total_huber_loss = total_consecutive_loss = total_softmean_loss = 0.0
    total_examples = total_steps = patience = 0

    batches = load_batches(
        bam_idx=bam_idx, 
        pod5_path=scope['train_pod5'], 
        batch_size=scope['batch_size'], 
        predict=False,
    )

    for batch, borders in batches:
        total_steps += 1
        print(f'Step {total_steps}')

        step_data = train_step(
            batch=batch, 
            labels=borders, 
            model=model, 
            device=device, 
            optimizer=optimizer,
            loss_f=loss_f, 
            total_steps=total_steps,
            scope=scope,
        )

        train_predictions = step_data['predictions']

        total_examples += len(batch)
        total_loss += step_data['loss']
        total_bce_loss += step_data['bce_loss']
        total_huber_loss += step_data['huber_loss']
        total_consecutive_loss += step_data['consecutive_loss']
        total_softmean_loss += step_data['softmean_loss']

        if (total_steps + 1) % scope['log_interval'] == 0:

            test_report = test_model(
                bam_idx=bam_idx, 
                model=model, 
                device=device, 
                loss_f=loss_f,
                scope=scope, 
                valid=True,
            )

            val_loss = test_report['loss']
            val_bce_loss = test_report['bce_loss']
            val_huber_loss = test_report['huber_loss']
            val_consecutive_loss = test_report['consecutive_loss']
            val_softmean_loss = test_report['softmean_loss']

            #print(f'Validation loss after {total_steps+1} is {val_loss:.2f}, BCE is {val_bce_loss:.2f}, Huber is {val_huber_loss:.2f}, Soft mean is {val_softmean_loss:.2f}', flush=True)

            if best_val_loss is None:
                best_val_loss = val_loss
                print('Saving first version of the model...')
                torch.save(model.state_dict(), scope['save_model'])
                print('Model saved.')

            if val_loss < best_val_loss:
                print(f'Saving new version of model with the lowest validation loss: {val_loss} < {best_val_loss}...')
                torch.save(model.state_dict(), scope['save_model'])
                print('Model saved.')
                best_val_loss = val_loss

            else: patience += 1

            #wandb.log({"epoch": epoch, "step": total_steps, "avg_train_loss": total_loss / total_steps, "avg_train_bce_loss": total_bce_loss / total_steps, "avg_train_huber_loss": total_huber_loss / total_steps, "avg_train_consecutive_loss": total_consecutive_loss / total_steps, "avg_train_softmean_loss": total_softmean_loss / total_steps,  "val_loss": val_loss, "val_bce_loss": val_bce_loss, "val_huber_loss": val_huber_loss, "val_consecutive_loss": val_consecutive_loss, "val_softmean_loss": val_softmean_loss})

    # log results after training for one epoch
    total_loss = total_loss.item()
    avg_loss = total_loss / total_steps

    test_report = test_model(
        bam_idx=bam_idx, 
        model=model, 
        device=device, 
        loss_f=loss_f, 
        scope=scope, 
        valid=True
    )

    val_loss, val_bce_loss, val_huber_loss, val_consecutive_loss, val_softmean_loss, _ = test_report
    print(f'Validation loss after epoch {epoch} is {val_loss}, BCE is {val_bce_loss}, Huber is {val_huber_loss}, consecutive {val_consecutive_loss}, soft mean {val_softmean_loss}')

    print(f'Train loss after epoch {epoch} is {total_loss / total_steps}, BCE is {total_bce_loss / total_steps}, Huber is {total_huber_loss / total_steps}, consecutive {total_consecutive_loss / total_steps}, soft mean {total_softmean_loss / total_steps}')

    if epoch >= new_loss_step:

        if best_val_loss is None:
                    best_val_loss = val_loss
                    print('Saving first version of the model')
                    torch.save(model.state_dict(), scope['save_model'])

        if val_loss < best_val_loss:
                    print(f'Saving new version of model with the lowest validation loss: {val_loss} < {best_val_loss}')
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), scope['save_model'])

    if epoch % 5 == 0:
        borders = torch.Tensor(borders).detach().cpu()
        predictions = torch.Tensor(train_predictions).detach().cpu()
        print(f'Epoch: {epoch}')
        print(f'Predicted: {torch.sum(torch.where(predictions > 0, 1, 0), dim=1)}')
        print(f'Labels: {torch.sum(borders, dim=1)}')
        print(f'Correct positions {torch.sum(torch.where(predictions > 0, 1, 0)*borders, dim=1)}')
        print(f'True logits: {torch.sum(predictions*borders, dim=1)}')
        print(f'False logits: {torch.sum(predictions*(1-borders), dim=1)}')

    wandb.log(
        {"epoch": epoch, "step": total_steps, "avg_train_loss": total_loss / total_steps,
         "avg_train_bce_loss": total_bce_loss / total_steps, "avg_train_huber_loss": total_huber_loss / total_steps,
         "avg_train_consecutive_loss": total_consecutive_loss / total_steps,
         "avg_train_softmean_loss": total_softmean_loss / total_steps, "val_loss": val_loss,
         "val_bce_loss": val_bce_loss, "val_huber_loss": val_huber_loss,
         "val_consecutive_loss": val_consecutive_loss, "val_softmean_loss": val_softmean_loss})

    return avg_loss, best_val_loss

def train_step(
        batch, 
        labels, 
        model: EventDetector, 
        device: torch.device, 
        loss_f: CustomLoss, 
        optimizer: torch.optim.Optimizer, 
        total_steps: int,
        scope: dict,
    ) -> dict:
    """
    Train model on a single batch.
    """
    report = dict()

    model.train()
    #batch = torch.unsqueeze(torch.Tensor(batch).to(device), 1)  #TODO when do I normalize the signal
    batch, labels = torch.Tensor(batch).to(device), torch.Tensor(labels).to(device)

    predictions = torch.squeeze(model(batch), dim=2)

    loss, bce_loss, huber_loss, consecutive_loss, softmean_loss = loss_f(batch, predictions, labels)

    if (total_steps + 1) % scope['log_interval'] == 0:
        print(f'Step {total_steps+1} Loss: {loss.item()}', flush=True)
        #num_predicted_events = torch.sum(torch.where(torch.sigmoid(torch.squeeze(predictions)) > 0.5, torch.tensor(1), torch.tensor(0)), dim=1).int()
        #num_true_events = torch.sum(labels, dim=1).int()
        #print(f'Num predicted vs true num events:\n\t{num_predicted_events[:10]}\n\t{num_true_events[:10]}, '
                   #f'alpha = {loss_f.alpha}, beta = {loss_f.beta}, gamma = {loss_f.gamma}')

    if torch.isnan(loss):
        print(batch)
        print(labels)
        print(torch.sum(labels, dim=1))
        print(predictions)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    report['loss'] = loss.item()
    report['bce_loss'] = bce_loss.item()
    report['huber_loss'] = huber_loss.item()
    report['consecutive_loss'] = consecutive_loss.item()
    report['softmean_loss'] = softmean_loss.item()
    report['predictions'] = predictions.detach()

    return report

def test_model(
          bam_idx: BamIndex, 
          model: EventDetector, 
          device: torch.device, 
          loss_f: CustomLoss, 
          scope: dict, 
          valid: bool = False
    ) -> dict:

    model.eval()

    full_predictions = []; full_labels = []

    # init losses
    full_loss = full_bce = full_huber = full_consecutive = full_softmean = 0.0

    steps = 0

    with torch.no_grad():

        batches = load_batches(bam_idx, scope['validation_pod5'], scope['val_batch_size'])
        for batch, labels in batches:
            batch, labels = torch.Tensor(batch).to(device), torch.Tensor(labels).to(device)

            predictions = torch.squeeze(model(batch), dim=2)

            loss, bce, huber, consecutive, softmean = loss_f(batch, predictions, labels)
            full_loss        += loss.item()
            full_bce         += bce.item()
            full_huber       += huber.item()
            full_consecutive += consecutive.item()
            full_softmean    += softmean.item()
            steps += 1

            if not valid:
                predictions = np.where((1 / (1 + np.exp(-predictions.detach().cpu().numpy()))) > 0.5, 1, 0)
                full_predictions.extend(list(predictions))
                full_labels.extend(labels.cpu().numpy())

        report = {}
        report['loss'] = full_loss / steps
        report['bce_loss'] = full_bce / steps
        report['huber_loss'] = full_huber / steps
        report['consecutive_loss'] = full_consecutive / steps
        report['softmean_loss'] = full_softmean / steps
        report['predictions'] = full_predictions

        return report
