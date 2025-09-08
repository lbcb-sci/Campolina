import torch
from torch import nn

class Default(nn.Module):
    '''
    Default model of Campolina paper: https://www.biorxiv.org/content/10.1101/2025.07.08.663658v1.
    '''
    name = 'Default'

    def __init__(
            self, 
            in_channels: int,
            out_channels: list, 
            classification_head: list, 
            kernel_size_one: int = 3, 
            kernel_size_all: int = 31, 
            stride_one: int = 1, 
            stride: int = 1, 
            dilations: list[int] = None,
            dropout_p: float = 0.1,
        ):

        super().__init__()

        if dilations is None: dilations = [1] * len(out_channels)
        assert len(dilations) == len(out_channels)

        layers = []
        layers.append(nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels[0], 
            kernel_size=kernel_size_one,
            stride=stride_one, 
            dilation=dilations[0],
            padding='same',
            padding_mode='zeros',
        ))
        layers.append(nn.BatchNorm1d(out_channels[0]))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(p=dropout_p))

        for i in range(1, len(out_channels)):
            layers.append(nn.Conv1d(
                in_channels=out_channels[i - 1], 
                out_channels=out_channels[i], 
                kernel_size=kernel_size_all,
                stride=stride, 
                dilation=dilations[i],
                padding='same', 
                padding_mode='zeros',
            ))

            layers.append(nn.BatchNorm1d(out_channels[i]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout_p))

        self.module_list = nn.ModuleList(layers)

        self.classification_head = nn.Sequential(nn.Linear(classification_head[0], classification_head[1]))
        for i in range(1, len(classification_head) - 1):
            self.classification_head.append(nn.GELU())
            self.classification_head.append(nn.Linear(classification_head[i], classification_head[i+1]))

    @classmethod
    def make_default(cls):
        return cls(
            in_channels=4, 
            out_channels=[32, 64, 64, 128, 128],
            classification_head=[128, 1], 
            kernel_size_one=3, 
            kernel_size_all=31,
        )

    def forward(self, x):
        for layer in self.module_list: x = layer.forward(x)
        x = torch.swapaxes(x, 1, 2)
        for layer in self.classification_head: x = layer(x)
        return x
