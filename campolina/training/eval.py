import time
import logging

import torch
from torch import nn, no_grad
from torcheval.metrics import BinaryF1Score

from campolina.data import BamIndex, load_batches_mp, DONE_SIGNAL
from campolina.loss import CustomLoss
from campolina.model import UNet

@no_grad()
def eval_model(
        bam_index: BamIndex,
        model: nn.Module, 
        device: torch.device, 
        loss_f: CustomLoss, 
        scope: dict, 
    ) -> dict:

    model.eval()

    all_probabilities = []; all_labels = []

    # init losses
    sumloss = sumfocal = sumhuber = sumconsec = 0.0
    steps = 0

    logger = logging.getLogger('eval')
    logger.info('starting model evaluation...')
    start = time.time()

    nprocesses = scope['nprocesses']

    processes, batches = load_batches_mp( 
        bam_index,
        pod5_path=scope['validation_pod5'], 
        batch_size=scope['val_batch_size'],
        nprocesses=nprocesses,
        length=scope['val_len'],
    )

    batch_size = None

    done_signals = 0
    while done_signals < nprocesses:

        try: 
            batch, labels = batches.get(timeout=5)
            if batch_size is None: batch_size = batch.shape[0]
        except:
            logger.warning(f'timeout (done signals = {done_signals}, nprocesses = {nprocesses})')
            continue

        if isinstance(batch, str) and batch == DONE_SIGNAL:
            done_signals += 1
            logger.info(f'received done signal (done signals = {done_signals}/{nprocesses})')
            continue

        batch, labels = batch.to(device), labels.to(device)

        predictions = model(batch)

        loss, focal, huber, consec = loss_f(predictions, labels)

        sumloss += loss.item(); sumfocal += focal.item()
        sumhuber += huber.item(); sumconsec += consec.item()
        steps += 1

        if batch.shape[0] != batch_size: continue

        all_probabilities.append(predictions.sigmoid())
        all_labels.append(labels)

    logger.info('joining processes...')
    for process in processes:
        try: process.join(timeout=10)
        except:
            logger.error(f'process {process.pid} timeout during join, terminating')
            process.terminate()

    logger.info('computing metrics...')

    probabilities = torch.stack(all_probabilities).flatten().to(device)
    labels = torch.stack(all_labels).flatten().to(device)

    # compute f1 on gpu if possible
    f1 = BinaryF1Score(device=device).update(probabilities, labels).compute()

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
