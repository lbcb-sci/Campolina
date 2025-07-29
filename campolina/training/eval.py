import time, logging
import torch
from torch import nn
import numpy as np
from sklearn.metrics import f1_score

from campolina.data import BamIndex, load_batches_mp, DONE_SIGNAL
from campolina.loss import CustomLoss
from campolina.model import UNet

@torch.no_grad
def eval_model(
        bam_index: BamIndex,
        model: nn.Module, 
        device: torch.device, 
        loss_f: CustomLoss, 
        scope: dict, 
        nprocesses: int,
    ) -> dict:

    model = model.eval()
    full_probabilities = []; full_labels = []

    # init losses
    sumloss = sumfocal = sumhuber = sumconsec = 0.0
    steps = 0

    logger = logging.getLogger('eval')
    logger.info('starting model evaluation...')
    start = time.time()

    processes, batches = load_batches_mp( 
        bam_index,
        pod5_path=scope['validation_pod5'], 
        #pod5_path=scope['train_pod5'], 
        batch_size=scope['val_batch_size'],
    )

    done_signals = 0
    while done_signals < nprocesses:

        try: batch, labels = batches.get(timeout=5)
        except:
            logger.warning(f'timeout (done signals = {done_signals}, nprocesses = {nprocesses})')
            continue

        if isinstance(batch, str) and batch == DONE_SIGNAL:
            done_signals += 1
            logger.info(f'received done signal (done signals = {done_signals})')
            continue

        batch, labels = batch.to(device), labels.to(device)
        batch = batch[:, :4, :] # remove t-statistic channel

        predictions = torch.squeeze(model(batch), dim=(1 if model.name == UNet.name else 2))

        loss, focal, huber, consec = loss_f(batch, predictions, labels)

        sumloss += loss.item(); sumfocal += focal.item()
        sumhuber += huber.item(); sumconsec += consec.item()
        steps += 1

        probabilities = predictions.sigmoid()
        full_probabilities.extend(list(probabilities.cpu().numpy()))
        full_labels.extend(list(labels.cpu().numpy()))

    logger.info('joining processes...')
    for process in processes:
        try: process.join(timeout=10)
        except:
            logger.error(f'process {process.pid} timeout during join, terminating')
            process.terminate()

    probabilities = np.array(full_probabilities)
    labels = np.array(full_labels)
    f1 = f1_score(labels.reshape(-1).astype(int), (probabilities.reshape(-1) > 0.5).astype(int))

    end = time.time()
    runtime = int(end - start)
    logger.info(f'model evaluation done. ({runtime} seconds)')

    return {
        'loss': sumloss / steps,
        'focal_loss': sumfocal / steps,
        'huber_loss': sumhuber / steps,
        'consecutive_loss': sumconsec / steps,
        'probabilities': probabilities,
        'labels': labels,
        'f1': f1,
    }