import time, os, logging
import torch
import torch.multiprocessing.queue

from campolina.model import EventDetector
from campolina.model.unet import UNet, make_big, make_medium, make_small
from campolina.loss import CustomLoss
from campolina.data import BamIndex, load_batches_mp, DONE_SIGNAL

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(scope: dict) -> None:
    '''
    Main function for training. Initializes model, loss and optim, and calls `train_epoch`.
    '''
    torch.multiprocessing.set_start_method('spawn')
    device = scope['devices'][0]
    logger = logging.getLogger('train'); logger.setLevel(logging.INFO)

    #model = EventDetector(
        #in_channels=scope['in_channels'], 
        #out_channels=scope['out_channels'], 
        #classification_head=scope['classification_head'], 
        #kernel_size_one=scope['kernel_one'], 
        #kernel_size_all=scope['kernel_all'],
        #dilation=scope['dilation'],
    #).to(device)

    logger.info('initializing model...')
    model = make_small(dropout=0.2).to(device)
    print(model)

    logger.info(f'Model number of parameters: {count_params(model)}')

    logger.info('torch.compile(model)...')
    model = torch.compile(model, fullgraph=True, backend='inductor')
    #model = torch.compile(model, fullgraph=True, backend='cudagraphs')

    logger.info('initializing loss function...')
    loss_f = CustomLoss.from_dict(scope)

    logger.info('initializing optimizer...')
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=scope['lr'], 
        eps=scope['adam_epsilon'],
    )

    bam_index = BamIndex(scope['bam_file'])

    best_val_loss = None
    epochs = scope['epochs']

    start_total = time.time()

    for epoch in range(1, epochs+1):
        logger.info(f'|| STARTING EPOCH {epoch} ||')

        start = time.time()

        train_epoch(
            bam_index=bam_index,
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
        logger.info(f'[[epoch {epoch} took {runtime / 60:.1f} minutes]]')

    end_total = time.time()
    total_runtime = int(end_total - start_total) / 60
    logger.info(f'training completed ({epochs} epochs) in {total_runtime:.1f} minutes.')

def train_epoch(
        model: EventDetector, 
        bam_index: BamIndex,
        device: torch.device, 
        optimizer: torch.optim.Optimizer, 
        loss_f: CustomLoss,
        scope: dict, 
        epoch: int, 
        best_val_loss: float, 
        nprocesses: int = 3,
    ):
    """
    Train the model for one epoch.
    """
    model.train()

    logger = logging.getLogger('train_epoch')

    batches: torch.multiprocessing.Queue
    processes: list[torch.multiprocessing.Process]
    processes, batches = load_batches_mp(
        bam_index,
        scope['train_pod5'],
        batch_size=scope['batch_size'],
        nprocesses=scope['nprocesses'],
    )

    time.sleep(2)
    total_steps = 0

    log_interval = scope['log_interval']
    running_loss = 0.0

    done_signals = 0
    while done_signals < nprocesses:

        try: batch, borders = batches.get(timeout=5)
        except:
            logger.warning(f'timeout but processes are not done (done signals = {done_signals} < {nprocesses})')
            continue

        if isinstance(batch, str) and batch == DONE_SIGNAL:
            done_signals += 1
            continue

        total_steps += 1

        train_report = train_step(
            batch=batch, 
            labels=borders, 
            model=model, 
            device=device, 
            optimizer=optimizer,
            loss_f=loss_f, 
        )

        running_loss += train_report["loss"]

        if total_steps % log_interval == 0:
            running_loss /= log_interval
            qstate = "full" if batches.full() else batches.qsize()
            logger.info(f'epoch {epoch}, step {total_steps}, loss = {running_loss:.4f}, queue~{qstate}')
            running_loss = 0.0

        if total_steps % scope['eval_interval'] == 0:
            test_report = eval_model(
                bam_index,
                model=model, 
                device=device, 
                loss_f=loss_f,
                scope=scope, 
            )

            print_eval(epoch, total_steps, test_report)

            val_loss = test_report['loss']

            logger.info('saving  model...')
            save_model(model, epoch, total_steps, val_loss)
            logger.info('model saved')

            #if best_val_loss is None:
                #best_val_loss = val_loss
                #logger.info('saving first version of the model...')
                #save_model(model, epoch, total_steps, val_loss)
                #logger.info('model saved')

            #elif val_loss < best_val_loss:
                #logger.info(f'saving new version of model with the lowest validation loss: {val_loss:.4f} < {best_val_loss:.4f}...')
                #save_model(model, epoch, total_steps, val_loss)
                #logger.info('model saved')
                #best_val_loss = val_loss

    logger.info('joining processes...')
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            logger.warning(f"process {p.pid} did not terminate. killing it.")
            p.terminate()

    logger.info(f'epoch {epoch} completed.')

def train_step(
        batch: torch.Tensor, 
        labels: torch.Tensor, 
        model: EventDetector, 
        device: torch.device, 
        loss_f: CustomLoss, 
        optimizer: torch.optim.Optimizer, 
    ) -> dict:
    """
    Train model on a single batch.
    """
    model.train()

    batch, labels = batch.to(device), labels.to(device)
    #predictions = torch.squeeze(model(batch), dim=2)
    #predictions = model(batch)
    if model.name == 'EventDetector':
        predictions = torch.squeeze(model(batch), dim=2)
    else:
        predictions = torch.squeeze(model(batch), dim=1)

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
        bam_index: BamIndex,
        model: EventDetector, 
        device: torch.device, 
        loss_f: CustomLoss, 
        scope: dict, 
    ) -> dict:
    start = time.time()
    model.eval()

    full_predictions = []; full_labels = []

    # init losses
    sumloss = sumfocal = sumhuber = sumconsec = 0.0
    steps = 0

    logger = logging.getLogger('eval')
    logger.info('starting model evaluation...')

    nprocesses = scope['nprocesses']

    processes, batches = load_batches_mp( 
        bam_index,
        pod5_path=scope['validation_pod5'], 
        batch_size=scope['val_batch_size'],
        nprocesses=nprocesses,
    )

    ones = 0
    true_positives = 0
    acc = 0.0
    i = 0

    done_signals = 0
    while done_signals < nprocesses:

        try: batch, labels = batches.get(timeout=5)
        except:
            logger.warning(f'timeout but processes are not done (done signals = {done_signals} < {nprocesses})')
            continue

        if isinstance(batch, str) and batch == DONE_SIGNAL:
            done_signals += 1
            continue

        batch, labels = batch.to(device), labels.to(device)

        with torch.no_grad():
            if model.name == 'EventDetector':
                predictions = torch.squeeze(model(batch), dim=2)
            else:
                predictions = torch.squeeze(model(batch), dim=1)

            loss, focal, huber, consec = loss_f(batch, predictions, labels)

            probabilities = predictions.sigmoid()
            preds = probabilities > 0.5

            tp = ((preds == 1) & (labels == 1)).sum().item() / (labels == 1).sum().item()
            true_positives += tp

            #sumacc = 0
            #for i, p in enumerate(preds):
                #sumacc += (p == labels[i]).int().mean()
            #sumacc /= labels.shape[0]
            #print(sumacc)
            ones += preds.sum().int()
            acc += (preds == labels).float().mean()
            i += 1
            #print((preds == labels).int().mean())

        sumloss += loss.item()
        sumfocal += focal.item()
        sumhuber += huber.item()
        sumconsec += consec.item()
        steps += 1

    logger.info(f'\naccuracy: {acc / float(i):.2f}')
    logger.info(f'ones: {ones:.2f}')
    logger.info(f'tp: {true_positives:.2f}\n')

    logger.info('joining processes...')
    #[process.join() for process in processes]
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            logger.warning(f"process {p.pid} did not terminate. killing it.")
            p.terminate()

    end = time.time()
    runtime = int(end - start)
    logger.info(f'model evaluation done. ({runtime} seconds)')

    return {
        'loss': sumloss / steps,
        'focal_loss': sumfocal / steps,
        'huber_loss': sumhuber / steps,
        'consecutive_loss': sumconsec / steps,
        'predictions': full_predictions,
    }

def mkname(epoch: int, step: int, loss: float):
    return f'epoch[{epoch}]_step[{step}]_loss[{loss:.4f}].pth'

def save_model(model: EventDetector, epoch: int, step: int, loss: float):
    path = f'models/{model.name}'
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f'{path}/{mkname(epoch, step, loss)}')

def print_eval(epoch: int, steps: int, report: dict):
    print(f'\nEvaluation @ epoch {epoch} @ step {steps}:')
    print(f'-- Total  Loss {report["loss"]:.4f}')
    print(f'-- Focal  Loss {report["focal_loss"]:.4f}')
    print(f'-- Huber  Loss {report["huber_loss"]:.4f}')
    print(f'-- Consec Loss {report["consecutive_loss"]:.4f}')
    print('', flush=True)
