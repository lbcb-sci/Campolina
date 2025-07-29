import os
import torch
from torch import nn

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mkname(epoch: int, step: int, val_loss: float, val_f1: float):
    return f'epoch[{epoch}]_step[{step}]_loss[{val_loss:.3f}]_f1[{val_f1:.3f}].pth'

def save_model(model: nn.Module, epoch: int, step: int, val_loss: float, val_f1: float, name: str = None):
    path = f'models/{model.name if name is None else name}'
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f'{path}/{mkname(epoch, step, val_loss, val_f1)}')

def print_eval(epoch: int, steps: int, report: dict):
    print(f'\nEvaluation @ epoch {epoch} @ step {steps}:')
    print(f'-- Total  Loss {report["loss"]:.4f}')
    print(f'-- Focal  Loss {report["focal_loss"]:.4f}')
    print(f'-- Huber  Loss {report["huber_loss"]:.4f}')
    print(f'-- Consec Loss {report["consecutive_loss"]:.4f}')
    print(f'-- F1 Score    {report["f1"]:.4f}')
    print('', flush=True)
