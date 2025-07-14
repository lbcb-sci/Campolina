import argparse, os, time, json
import wandb
import numpy as np
import pod5 as p5
from tqdm import tqdm

import torch
from torch.optim import AdamW

from campolina.data.bam_util import BamIndex
from campolina.data.pod5_util import get_reads, process_chunk
from campolina.model.model import EventDetector
from loss import CustomLoss

#torch.manual_seed(12345)

def load_batches(bam_idx: BamIndex, pod5_path: str, batch_size: int, predict: bool = False): 
    """
    Load data for training or validation.
    """

    # TODO what if i preload a batch of batch_size signals and then yield batches as long as i can and then load batch_size signal again?
    current_batch = []; current_borders = []; current_identifiers = []

    for read in get_reads(pod5_path):
        for alignment in bam_idx.get_alignment(str(read.read_id)):
            if alignment is None: continue

            signal_chunks, chunk_borders, chunk_identifiers = process_chunk(
                aln=alignment, 
                read=read, 
                predict=predict, 
                adjust_type=None,
            )

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

                if predict: current_identifiers = chunk_identifiers[to_take:]

            else:
                current_batch.extend(signal_chunks)
                current_borders.extend(chunk_borders)
                if predict: current_identifiers.extend(chunk_identifiers)

    if predict:
        yield np.array(current_batch), np.array(current_borders), np.array(current_identifiers)
    else:
        #print(read.read_id)
        yield np.array(current_batch), np.array(current_borders)

def test_model(bam_idx: BamIndex, model: EventDetector, device, loss_f, scope: dict, valid: bool = False):
    model.eval()

    full_predictions = []; full_labels = []
    full_loss = full_bce_loss = full_huber_loss = full_consecutive_loss = full_softmean_loss = 0.0
    steps = 0

    with torch.no_grad():
        batches = load_batches(bam_idx, scope['validation_pod5'], scope['val_batch_size'])

        for batch, labels in tqdm(batches):
            batch = torch.Tensor(batch).to(device)
            labels = torch.Tensor(labels).to(device)

            predictions = torch.squeeze(model(batch), dim=2)

            loss, bce_loss, huber_loss, consecutive_loss, softmean_loss = loss_f(batch, predictions, labels)
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

    report = {}
    report['loss'] = full_loss / steps
    report['bce_loss'] = full_bce_loss / steps
    report['huber_loss'] = full_huber_loss / steps
    report['consecutive_loss'] = full_consecutive_loss / steps
    report['softmean_loss'] = full_softmean_loss / steps
    report['predictions'] = full_predictions

    return report

def train_step(
        batch, 
        labels, 
        model, 
        device, 
        loss_f, 
        optimizer, 
        total_steps
    ):

    report = dict()

    model.train()
    #batch = torch.unsqueeze(torch.Tensor(batch).to(device), 1)  #TODO when do I normalize the signal
    batch = torch.Tensor(batch).to(device)
    labels = torch.Tensor(labels).to(device)

    predictions = torch.squeeze(model(batch), dim=2)
    report['predictions'] = predictions

    loss, bce_loss, huber_loss, consecutive_loss, softmean_loss = loss_f(batch, predictions, labels)

    report['loss'] = loss
    report['bce_loss'] = bce_loss
    report['huber_loss'] = huber_loss
    report['consecutive_loss'] = consecutive_loss
    report['softmean_loss'] = softmean_loss

    if total_steps % 3000 == 0:
        num_predicted_events = torch.sum(torch.where(torch.sigmoid(torch.squeeze(predictions)) > 0.5, torch.tensor(1), torch.tensor(0)), dim=1).int()
        num_true_events = torch.sum(labels, dim=1).int()
        tqdm.write(f'Num predicted vs true num events:\n\t{num_predicted_events[:10]}\n\t{num_true_events[:10]}, '
                   f'alpha = {loss_f.alpha}, beta = {loss_f.beta}, gamma = {loss_f.gamma}')

    if torch.isnan(loss):
        print(batch)
        print(labels)
        print(torch.sum(labels, dim=1))
        print(predictions)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return report

def train_epoch(
        bam_idx: BamIndex, 
        model: EventDetector, 
        device, 
        optimizer: torch.optim.Optimizer, 
        loss_f, 
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

    for batch, borders in tqdm(batches):
        total_steps += 1

        step_data = train_step(
            batch=batch, 
            labels=borders, 
            model=model, 
            device=device, 
            optimizer=optimizer,
            loss_f=loss_f, 
            total_steps=total_steps,
        )

        train_predictions = step_data['predictions']

        total_examples += len(batch)
        total_loss += step_data['loss'].detach()
        total_bce_loss += step_data['bce_loss'].detach()
        total_huber_loss += step_data['huber_loss'].detach()
        total_consecutive_loss += step_data['consecutive_loss'].detach()
        total_softmean_loss += step_data['softmean_loss'].detach()

        if total_steps % 3000 == 0:

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

            tqdm.write(f'Validation loss after {total_steps} is {val_loss:.2f}, BCE is {val_bce_loss:.2f}, Huber is {val_huber_loss:.2f}, Soft mean is {val_softmean_loss:.2f}')

            if best_val_loss is None:
                best_val_loss = val_loss
                tqdm.write('Saving first version of the model...')
                torch.save(model.state_dict(), scope['save_model'])
                tqdm.write('Model saved.')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                tqdm.write(f'Saving new version of model with the lowest validation loss: {val_loss} < {best_val_loss}...')
                torch.save(model.state_dict(), scope['save_model'])
                tqdm.write('Model saved.')

            else: patience += 1

            #wandb.log(
                    #{"epoch": i, "step": total_steps, "avg_train_loss": total_loss / total_steps, "avg_train_bce_loss": total_bce_loss / total_steps, "avg_train_huber_loss": total_huber_loss / total_steps, "avg_train_consecutive_loss": total_consecutive_loss / total_steps, "avg_train_softmean_loss": total_softmean_loss / total_steps,  "val_loss": validation_loss, "val_bce_loss": validation_bce_loss, "val_huber_loss": validation_huber_loss, "val_consecutive_loss": validation_consecutive_loss, "val_softmean_loss": validation_softmean_loss})

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
    tqdm.write(f'Validation loss after epoch {epoch} is {val_loss}, BCE is {val_bce_loss}, Huber is {val_huber_loss}, consecutive {val_consecutive_loss}, soft mean {val_softmean_loss}')

    tqdm.write(f'Train loss after epoch {epoch} is {total_loss / total_steps}, BCE is {total_bce_loss / total_steps}, Huber is {total_huber_loss / total_steps}, consecutive {total_consecutive_loss / total_steps}, soft mean {total_softmean_loss / total_steps}')

    if epoch >= new_loss_step:

        if best_val_loss is None:
                    best_val_loss = val_loss
                    tqdm.write('Saving first version of the model')
                    torch.save(model.state_dict(), scope['save_model'])

        if val_loss < best_val_loss:
                    tqdm.write(f'Saving new version of model with the lowest validation loss: {val_loss} < {best_val_loss}')
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

    #wandb.log(
        #{"epoch": i, "step": total_steps, "avg_train_loss": total_loss / total_steps,
         #"avg_train_bce_loss": total_bce_loss / total_steps, "avg_train_huber_loss": total_huber_loss / total_steps,
         #"avg_train_consecutive_loss": total_consecutive_loss / total_steps,
         #"avg_train_softmean_loss": total_softmean_loss / total_steps, "val_loss": validation_loss,
         #"val_bce_loss": validation_bce_loss, "val_huber_loss": validation_huber_loss,
         #"val_consecutive_loss": validation_consecutive_loss, "val_softmean_loss": validation_softmean_loss})

    return avg_loss, best_val_loss

def train(scope: dict):
    device = scope['devices'][0]

    #wandb.init(project="event_detecting", entity="bakicsara97")

    bam_idx = BamIndex(scope['bam_file'])

    # init model
    model = EventDetector(
        in_channels=scope['in_channels'], 
        out_channels=scope['out_channels'], 
        classification_head=scope['classification_head'], 
        kernel_size_one=scope['kernel_one'], 
        kernel_size_all=scope['kernel_all'],
    ).to(device)

    #wandb.watch(model)

    # init custom loss function
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

    optimizer = AdamW(
        model.parameters(), 
        lr=scope['lr'], 
        eps=scope['adam_epsilon'],
    )

    best_val_loss = None

    for i in range(scope['epochs']):
        tqdm.write(f'Starting epoch {i}')

        train_loss, best_val_loss = train_epoch(
            bam_idx=bam_idx, 
            model=model, 
            device=device, 
            optimizer=optimizer, 
            loss_f=loss_f, 
            scope=scope, 
            best_val_loss=best_val_loss, 
            epoch=i, 
            new_loss_step=scope['introduce_losses']
        ) # TODO myb do not evaluate on padded part of the signal

    print(f'Model saved to {scope["save_model"]}')
    #wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='train_config.json')

    args = parser.parse_args()
    with open(args.config_file, 'r') as inf: scope = json.load(inf)

    if 'gpu' in scope and len(scope['gpu']) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join((str(x) for x in scope['gpu']))
        scope["devices"] = [torch.device("cuda", x) for x in range(len(scope['gpu']))]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        scope["devices"] = [torch.device("cpu")]
    
    print(f'Using {scope["devices"]}')

    train(scope) # main function