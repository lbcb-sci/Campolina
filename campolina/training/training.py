import time, logging
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchscan import summary

from campolina.data import BamIndex, load_batches_mp, DONE_SIGNAL
from campolina.model import Default, unet
from campolina.loss import CustomLoss

from .utils import save_model, print_eval, count_params
from .eval import eval_model

def train(scope: dict, run_name: str = 'model') -> None:
    '''
    Main function for training. Initializes model, loss and optim, and calls `train_epoch`.
    '''
    torch.multiprocessing.set_start_method('spawn')
    device = scope['devices'][0]
    logger = logging.getLogger('train'); logger.setLevel(logging.INFO)

    logger.info('initializing model...')
    model = unet.make_default().to(device)
    #print(model)
    summary(model, (4, 6000))

    logger.info(f'model #parameters = {count_params(model):,d}')

    #logger.info('torch.compile(model)...')
    #model = torch.compile(model, fullgraph=True, backend='inductor')

    logger.info('initializing loss function...')
    loss_f = CustomLoss.from_dict(scope)

    logger.info('initializing optimizer...')
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=scope['lr'], 
        eps=scope['adam_epsilon'],
    )

    bam_index = BamIndex(scope['bam_file'])
    tensorboard = SummaryWriter(f'runs/{run_name}')
    epochs = scope['epochs']
    total_steps = 0
    start_total = time.time()

    for epoch in range(1, epochs+1):
        logger.info(f'|| STARTING EPOCH {epoch} ||')

        start_epoch = time.time()

        total_steps = train_epoch(
            bam_index=bam_index,
            model=model, 
            device=device, 
            optimizer=optimizer, 
            loss_f=loss_f, 
            scope=scope, 
            total_steps=total_steps,
            epoch=epoch, 
            tensorboard=tensorboard,
            run_name=run_name,
        )

        end_epoch = time.time()
        runtime_epoch = end_epoch - start_epoch
        logger.info(f'[[epoch {epoch} took {runtime_epoch / 60:.1f} minutes]]')

    end_total = time.time()
    total_runtime = int(end_total - start_total) / 60
    logger.info(f'training completed ({epochs} epochs) in {total_runtime:.1f} minutes.')
    tensorboard.close()

def train_epoch(
        model: Default, 
        bam_index: BamIndex,
        device: torch.device, 
        optimizer: torch.optim.Optimizer, 
        loss_f: CustomLoss,
        scope: dict, 
        epoch: int, 
        total_steps: int,
        tensorboard: SummaryWriter,
        run_name: str = None,
    ):
    """
    Train the model for a single epoch.
    """
    model.train()
    logger = logging.getLogger('train_epoch')
    nprocesses = scope['nprocesses']

    batches: torch.multiprocessing.Queue
    processes: list[torch.multiprocessing.Process]
    processes, batches = load_batches_mp(
        bam_index,
        scope['train_pod5'],
        batch_size=scope['batch_size'],
        nprocesses=nprocesses,
    )

    time.sleep(2)

    log_interval = scope['log_interval']
    running_loss = 0.0

    done_signals = 0
    while done_signals < nprocesses:

        try: batch, borders = batches.get(timeout=5)
        except:
            logger.warning(f'timeout (done signals = {done_signals}, nprocesses = {nprocesses})')
            continue

        if isinstance(batch, str) and batch == DONE_SIGNAL:
            done_signals += 1
            logger.info(f'received done signal (done signals = {done_signals}/{nprocesses})')
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

        train_loss = train_report['loss']

        tensorboard.add_scalar('train_loss', train_loss, total_steps)

        tensorboard.add_pr_curve(
            'P/R training',
            labels=train_report['labels'],
            predictions=train_report['probabilities'],
            global_step=total_steps,
        )

        running_loss += train_loss

        if total_steps % log_interval == 0:
            running_loss /= log_interval
            qstate = "full" if batches.full() else batches.qsize()
            logger.info(f'epoch {epoch}, step {total_steps}, loss = {running_loss:.4f}, queue~{qstate}')
            running_loss = 0.0
    
    logger.info('joining processes...')
    for process in processes:
        try: process.join(timeout=10)
        except:
            logger.error(f'process {process.pid} timeout during join, terminating')
            process.terminate()

    # eval at the end of each epoch
    val_report = eval_model(
        bam_index,
        model=model, 
        device=device, 
        loss_f=loss_f,
        scope=scope, 
        nprocesses=nprocesses,
    )

    print_eval(epoch, total_steps, val_report)

    val_loss = val_report['loss']
    tensorboard.add_scalar('val_loss', val_loss, total_steps)

    val_f1 = val_report['f1']
    tensorboard.add_scalar(tag='F1 (val)', scalar_value=val_f1, global_step=total_steps)

    tensorboard.add_pr_curve(
        'P/R validation',
        labels=torch.tensor(val_report['labels']),
        predictions=torch.tensor(val_report['probabilities']),
        global_step=total_steps,
    )

    logger.info('saving model...')
    save_model(
        model=model, 
        epoch=epoch, 
        step=total_steps, 
        val_loss=val_loss, 
        val_f1=val_f1, 
        name=run_name
    )

    logger.info('model saved')

    logger.info(f'epoch {epoch} completed.')
    return total_steps

def train_step(
        batch: torch.Tensor, 
        labels: torch.Tensor, 
        model: Default, 
        device: torch.device, 
        loss_f: CustomLoss, 
        optimizer: torch.optim.Optimizer, 
    ) -> dict:
    """
    Train model on a single batch, that is forward + backward pass, and returns loss details.
    """
    model.train()

    batch, labels = batch.to(device), labels.to(device)
    predictions = torch.squeeze(model(batch), dim=1 if model.name == unet.name else 2)

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
        'probabilities': (predictions.sigmoid()).detach(),
        'labels': labels.detach(),
    }
