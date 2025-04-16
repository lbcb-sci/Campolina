import torch
from torch import nn
from pytorch_tcn import TCN


class TCNEventDetector(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, classification_head, dilations=None, dilation_reset=16, dropout=0.1, causal=False, use_norm='weight_norm', activation='gelu'):
        super().__init__()
        #layers = []

        #for i in range(0, len(channels)-1):
        self.tcn_block = TCN(num_inputs=in_channels,
                            num_channels=channels,
                            kernel_size=kernel_size,
                            dilations=dilations,
                            dilation_reset=dilation_reset,
                            dropout=dropout,
                            causal=causal,
                            use_norm=use_norm,
                            activation=activation
                            )
        self.classification_head = nn.Sequential(nn.Linear(classification_head[0], classification_head[1]))
        for i in range(1, len(classification_head) - 1):
            self.classification_head.append(nn.GELU())
            self.classification_head.append(nn.Linear(classification_head[i], classification_head[i+1]))
        #self.classification_head = nn.Sequential(nn.Linear(channels[-1], 256), nn.ReLU(), nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 1))


    def forward(self, x):
        x = self.tcn_block(x)
        x = torch.swapaxes(x, 1, 2)
        x = self.classification_head(x)

        return x
