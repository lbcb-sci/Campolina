import os
import torch
from torch import nn

def mkname(epoch: int, step: int, val_loss: float):
    return f'epoch[{epoch}]_step[{step}]_loss[{val_loss:.4f}].pth'

def save_model(model: nn.Module, epoch: int, step: int, val_loss: float):
    path = f'models/{model.name}'
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f'{path}/{mkname(epoch, step, val_loss)}')

def print_eval(epoch: int, steps: int, report: dict):
    print(f'\nEvaluation @ epoch {epoch} @ step {steps}:')
    print(f'-- Total  Loss {report["loss"]:.4f}')
    print(f'-- Focal  Loss {report["focal_loss"]:.4f}')
    print(f'-- Huber  Loss {report["huber_loss"]:.4f}')
    print(f'-- Consec Loss {report["consecutive_loss"]:.4f}')
    print('', flush=True)
