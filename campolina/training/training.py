import time, os, logging
import wandb
import torch
import numpy as np

from campolina.model import EventDetector
from campolina.loss import CustomLoss
from campolina.data import BamIndex, load_batches

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

    logger.info('torch.compile(model)...')
    model = torch.compile(model, fullgraph=True, backend='inductor')

    wandb.watch(model)

    logger.info('initializing loss function...')
    loss_f = CustomLoss.from_dict(scope)

    logger.info('initializing optimizer...')
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=scope['lr'], 
        eps=scope['adam_epsilon'],
    )

    best_val_loss = None

    bam_idx = BamIndex(scope['bam_file'])

    for epoch in range(1, scope['epochs']+1):
        logger.info(f'[[starting epoch {epoch}...]]')

        start = time.time()
        train_epoch(
            bam_idx=bam_idx, 
            model=model, 
            device=device, 
            optimizer=optimizer, 
            loss_f=loss_f, 
            scope=scope, 
            best_val_loss=best_val_loss, 
            epoch=epoch, 
        ) # TODO myb do not evaluate on padded part of the signal

        end = time.time()
        runtime = end - start
        logger.info(f'[[epoch {epoch} took {runtime / 60} minutes]]')

    #logger.info(f'model saved to {scope["save_model"]}')
    wandb.finish()

def train_epoch(
        bam_idx: BamIndex, 
        model: EventDetector, 
        device: torch.device, 
        optimizer: torch.optim.Optimizer, 
        loss_f: CustomLoss,
        scope: dict, 
        best_val_loss: float, 
        epoch: int, 
    ):
    """
    Train the model for one epoch.
    """

    model.train()

    logger = logging.getLogger('train_epoch')

    batches = load_batches(
        bam_idx=bam_idx, 
        pod5_path=scope['train_pod5'], 
        batch_size=scope['batch_size'], 
        predict=False,
    )

    total_steps = patience = 0

    for batch, borders in batches:
        total_steps += 1

        train_report = train_step(
            batch=batch, 
            labels=borders, 
            model=model, 
            device=device, 
            optimizer=optimizer,
            loss_f=loss_f, 
        )

        logger.info(f'step {total_steps}, loss = {train_report["loss"]:.4f}')

        if total_steps % scope['eval_interval'] == 0:

            test_report = eval_model(
                bam_idx=bam_idx, 
                model=model, 
                device=device, 
                loss_f=loss_f,
                scope=scope, 
                valid=True,
            )

            print_eval(epoch, total_steps, test_report, patience)
            val_loss = test_report['loss']

            if best_val_loss is None:
                best_val_loss = val_loss
                logger.info('saving first version of the model...')
                save_model(model, epoch, total_steps, scope)
                logger.info('model saved')
                patience = 0

            elif val_loss < best_val_loss:
                logger.info(f'saving new version of model with the lowest val_loss: {val_loss:.4f} < {best_val_loss:.4f}...')
                save_model(model, epoch, total_steps, scope)
                logger.info('model saved')
                best_val_loss = val_loss
                patience = 0

            else: patience += 1

def train_step(
        batch, 
        labels, 
        model: EventDetector, 
        device: torch.device, 
        loss_f: CustomLoss, 
        optimizer: torch.optim.Optimizer, 
    ) -> dict:
    """
    Train model on a single batch.
    """
    model.train()

    batch, labels = torch.Tensor(batch).to(device), torch.Tensor(labels).to(device)
    predictions = torch.squeeze(model(batch), dim=2)

    loss, focal, huber, consec = loss_f(batch, predictions, labels)

    if torch.isnan(loss):
        logger = logging.getLogger('train_step')
        logger.error('loss function returned NaN')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        'loss': loss.item(),
        'focal_loss': focal.item(),
        'huber_loss': huber.item(),
        'consecutive_loss': consec.item(),
        'predictions': predictions.detach(),
    }

def eval_model(
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
    sumloss = sumfocal = sumhuber = sumconsec = 0.0
    steps = 0

    logger = logging.getLogger('eval')
    logger.info('starting model evaluation...')
    with torch.no_grad():
        batches = load_batches(bam_idx, scope['validation_pod5'], scope['val_batch_size'])
        for batch, labels in batches:
            batch, labels = torch.Tensor(batch).to(device), torch.Tensor(labels).to(device)

            predictions = torch.squeeze(model(batch), dim=2)

            loss, focal, huber, consec = loss_f(batch, predictions, labels)
            sumloss += loss.item()
            sumfocal += focal.item()
            sumhuber += huber.item()
            sumconsec += consec.item()
            steps += 1

            #if not valid: # TODO
                #predictions = np.where((1 / (1 + np.exp(-predictions.detach().cpu().numpy()))) > 0.5, 1, 0)
                #full_predictions.extend(list(predictions))
                #full_labels.extend(labels.cpu().numpy())

        logger.info('model evaluation done.')

        return {
            'loss': sumloss / steps,
            'focal_loss': sumfocal / steps,
            'huber_loss': sumhuber / steps,
            'consecutive_loss': sumconsec / steps,
            'predictions': full_predictions,
        }

def mkname(epoch: int, step: int, scope: dict):
    alpha, beta, gamma = scope['bce_alpha'], scope['huber_beta'], scope['consecutive_gamma']
    return f'epoch[{epoch}]_step[{step}]_alpha[{alpha}]_beta[{beta}]_gamma[{gamma}].pth'

def save_model(model: EventDetector, epoch: int, step: int, scope: dict):
    try: os.mkdir('models')
    finally: torch.save(model.state_dict(), f'models/{mkname(epoch, step, scope)}')

def print_eval(epoch: int, steps: int, report: dict, patience: int):
    print(f'\nEvaluation @ epoch {epoch} @ step {steps}:')
    print(f'-- Total    Loss {report["loss"]:.4f}')
    print(f'-- Focal    Loss {report["focal_loss"]:.4f}')
    print(f'-- Huber    Loss {report["huber_loss"]:.4f}')
    print(f'-- Consec   Loss {report["consecutive_loss"]:.4f}')
    print(f'-- Patience {patience}')
    print('', flush=True)
