import os
import torch
from torch import nn

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mkname(epoch: int, step: int, val_loss: float, val_f1: float):
    return f'epoch_{epoch}_step_{step}_loss_{val_loss:.3f}_f1_{val_f1:.3f}.pth'

def save_model(model: nn.Module, epoch: int, step: int, val_loss: float, val_f1: float, name: str = None):
    path = f'weights/{model.name if name is None else name}'
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f'{path}/{mkname(epoch, step, val_loss, val_f1)}')

def print_eval(epoch: int, steps: int, metrics: dict):
    print(f'\nEvaluation @ epoch {epoch} @ step {steps}:')
    print(f'-- Total  {metrics["loss"]:.4f}')
    print(f'-- Focal  {metrics["focal_loss"]:.4f}')
    print(f'-- Huber  {metrics["huber_loss"]:.4f}')
    print(f'-- Consec {metrics["consecutive_loss"]:.4f}')
    print(f'-- F1     {metrics["f1"]:.4f}')
    print('', flush=True)
