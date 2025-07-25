import time, logging
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from campolina.model import EventDetector, UNet
from campolina.loss import CustomLoss
from campolina.data import BamIndex, load_batches_mp, DONE_SIGNAL

from .eval import eval_model
from .utils import save_model, print_eval

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(scope: dict, run_name: str = 'unet') -> None:
    '''
    Main function for training. Initializes model, loss and optim, and calls `train_epoch`.
    '''
    torch.multiprocessing.set_start_method('spawn')
    device = scope['devices'][0]
    logger = logging.getLogger('train'); logger.setLevel(logging.INFO)

    #logger.info('initializing model...')
    #model = EventDetector(
        #in_channels=scope['in_channels'], 
        #out_channels=scope['out_channels'], 
        #classification_head=scope['classification_head'], 
        #kernel_size_one=scope['kernel_one'], 
        #kernel_size_all=scope['kernel_all'],
        #dilation=scope['dilation'],
    #).to(device)
    #print(model)

    logger.info('initializing model...')
    model = UNet.make_default(dropout=0.2).to(device)
    print(model)

    logger.info(f'number of parameters: {count_params(model)}')

    logger.info('torch.compile(model)...')
    model = torch.compile(model, fullgraph=True, backend='inductor')

    logger.info('initializing loss function...')
    loss_f = CustomLoss.from_dict(scope)

    logger.info('initializing optimizer...')
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=scope['lr'], 
        eps=scope['adam_epsilon'],
        weight_decay=0.01,
    )

    bam_index = BamIndex(scope['bam_file'])

    best_val_loss = None
    epochs = scope['epochs']

    start_total = time.time()
    tb = SummaryWriter(f'runs/{run_name}')

    total_steps = 0

    for epoch in range(1, epochs+1):
        logger.info(f'|| STARTING EPOCH {epoch} ||')

        start = time.time()

        total_steps = train_epoch(
            bam_index=bam_index,
            model=model, 
            device=device, 
            optimizer=optimizer, 
            loss_f=loss_f, 
            scope=scope, 
            total_steps=total_steps,
            best_val_loss=best_val_loss, 
            epoch=epoch, 
            tb=tb,
        ) # TODO myb do not evaluate on padded part of the signal

        end = time.time()
        runtime = end - start
        logger.info(f'[[epoch {epoch} took {runtime / 60:.1f} minutes]]')

    end_total = time.time()
    total_runtime = int(end_total - start_total) / 60
    logger.info(f'training completed ({epochs} epochs) in {total_runtime:.1f} minutes.')
    tb.close()

def train_epoch(
        model: EventDetector, 
        bam_index: BamIndex,
        device: torch.device, 
        optimizer: torch.optim.Optimizer, 
        loss_f: CustomLoss,
        scope: dict, 
        epoch: int, 
        total_steps: int,
        best_val_loss: float, 
        nprocesses: int = 3,
        tb = None,
    ):
    """
    Train the model for a single epoch.
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

    log_interval = scope['log_interval']
    running_loss = 0.0

    writer = SummaryWriter(log_dir='runs/dilations=1-1-3-5-7')

    done_signals = 0
    while done_signals < nprocesses:

        try: batch, borders = batches.get(timeout=5)
        except:
            logger.warning(f'timeout (done signals = {done_signals}, nprocesses = {nprocesses})')
            continue

        if isinstance(batch, str) and batch == DONE_SIGNAL:
            done_signals += 1
            logger.info(f'received done signal (done signals = {done_signals})')
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

        writer.add_scalar(
            tag='training loss', 
            scalar_value=train_report["loss"],
            global_step=epoch*total_steps,
        )

        writer.add_scalar(
            tag='true positives (train)', 
            scalar_value=train_report["true_positives"],
            global_step=epoch*total_steps,
        )

        writer.add_pr_curve(
            'precision / recall (train)',
            labels=train_report["labels"],
            predictions=train_report["probabilities"],
            global_step=epoch*total_steps,
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
                nprocesses=nprocesses,
            )

            print_eval(epoch, total_steps, test_report)

            val_loss = test_report['loss']
            writer.add_scalar(
                'validation loss', 
                test_report["loss"],
                epoch*total_steps,
            )

            writer.add_scalar(
                tag='true positives (val)', 
                scalar_value=test_report["true_positives"],
                global_step=epoch*total_steps,
            )

            writer.add_pr_curve(
                'precision / recall (val)',
                labels=torch.tensor(test_report["labels"]),
                predictions=torch.tensor(test_report["probabilities"]),
                global_step=epoch*total_steps,
            )

            logger.info('saving model...')
            save_model(model, epoch, total_steps, val_loss)
            logger.info('model saved')

            #if best_val_loss is None:
                #best_val_loss = val_loss
                #logger.info('saving first version of the model...')
                #save_model(model, epoch, total_steps, scope)
                #logger.info('model saved')

            #elif val_loss < best_val_loss:
                #logger.info(f'saving new version of model with the lowest validation loss: {val_loss:.4f} < {best_val_loss:.4f}...')
                #save_model(model, epoch, total_steps, scope)
                #logger.info('model saved')
                #best_val_loss = val_loss

    logger.info('joining processes...')
    for process in processes:
        try: process.join(timeout=10)
        except:
            logger.error(f'process {process.pid} timeout during join, terminating')
            process.terminate()

    logger.info(f'epoch {epoch} completed.')
    writer.close()

def train_step(
        batch: torch.Tensor, 
        labels: torch.Tensor, 
        model: EventDetector, 
        device: torch.device, 
        loss_f: CustomLoss, 
        optimizer: torch.optim.Optimizer, 
    ) -> dict:
    """
    Train model on a single batch, that is forward + backward pass, and returns loss details.
    """
    model.train()

    batch, labels = batch.to(device), labels.to(device)
    predictions = torch.squeeze(
        model(batch), 
        dim=2 if model.name == 'EventDetector' else 1,
    )

    loss, focal, huber, consec = loss_f(batch, predictions, labels)

    if torch.isnan(loss):
        logger = logging.getLogger('train_step')
        logger.error('loss function returned NaN')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    return {
        'loss': loss.item(),
        'focal_loss': focal.item(),
        'huber_loss': huber.item(),
        'consecutive_loss': consec.item(),
        #'probabilities': probabilities.detach(),
        'labels': labels.detach(),
        #'true_positives': true_positives,
    }
