import torch
from torch import nn, Tensor
from torch.utils import checkpoint

DEFAULT_KERNEL_SIZE = 3
DEFAULT_INPUT_CHANNELS = 4

class unet(nn.Module):
    '''
    (Currently being tested, see `Default` for the official Campolina model.)
    '''
    name = 'unet'

    def __init__(
            self,
            channels: list[int],
            n_channels_input: int = DEFAULT_INPUT_CHANNELS,
            kernel: int = DEFAULT_KERNEL_SIZE,
            dropout: float = 0.0,
            checkpoint: bool = True,
        ):
        super().__init__()
        rev_channels = list(reversed(channels))

        # dummy layer to map to same number of channels as final layer and normalize for downsampling
        self.first = nn.Sequential(
            nn.Conv1d(n_channels_input, channels[0], 1, 1, bias=False),
            nn.BatchNorm1d(num_features=channels[0]),
        ) 

        self.downlayers = nn.ModuleList([
            Down(
                in_ch=in_ch, 
                out_ch=out_ch,
                kernel=kernel,
                dropout=dropout,
            ) for (in_ch, out_ch) in zip(channels[:-1], channels[1:])
        ])

        self.middle = ConvBlock(channels[-1], channels[-1], 3, 0)

        self.uplayers = nn.ModuleList([
            Up(
                in_ch=in_ch*2, # + residual
                out_ch=out_ch,
                kernel=kernel,
                dropout=dropout,
                checkpoint=checkpoint,
            ) for (in_ch, out_ch) in zip(rev_channels[:-1], rev_channels[1:])
        ])

        self.last = nn.Linear(channels[0]*2, 1)

    @classmethod
    def make_default(cls, dropout: float = 0.0):
        chan = [16, 32, 64, 128, 256]
        return cls(channels=chan, dropout=dropout)
        
    def forward(self, tensor: Tensor) -> Tensor:
        tensor = self.first(tensor)
        first = tensor.clone()

        residuals = []

        # downsample
        for down in self.downlayers:
            tensor = down(tensor)
            residuals.append(tensor.clone())

        # bottleneck
        tensor = self.middle(tensor)

        # upsample
        for i, up in enumerate(self.uplayers):
            residual = residuals[-(i+1)]
            tensor = torch.concat([tensor, residual], dim=1)
            tensor = up(tensor)

        # get logits
        tensor = torch.concat([tensor, first], dim=1)
        return self.last(tensor.transpose(-2, -1)).squeeze(-1)

class Down(nn.Module):
    '''Halves the input by applying pooling + 2*conv.'''
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float):
        super().__init__()

        self.layers = nn.Sequential(
            DoubleConvBlock(in_ch, out_ch, kernel, dropout),
            nn.MaxPool1d(2, 2),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.layers(tensor)

class Up(nn.Module):
    '''Upsample the input using nearest upsampling + 2*conv.'''
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float, checkpoint: bool):
        super().__init__()
        assert kernel % 2 == 1
        self.ckpt = checkpoint

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            DoubleConvBlock(in_ch, out_ch, kernel, dropout)
        )

    def forward(self, tensor: Tensor) -> Tensor:
        if not self.ckpt: return self.layers(tensor)
        return checkpoint.checkpoint(lambda _: self.layers(_), tensor, use_reentrant=False)

class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float):
        super().__init__()
        mid_ch = max(in_ch, out_ch)

        self.layers = nn.Sequential(
            ConvBlock(in_ch, mid_ch, kernel, dropout),
            ConvBlock(mid_ch, out_ch, kernel, dropout),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.layers(tensor)

class ConvBlock(nn.Module):
    '''Conv + Norm + ReLU + Dropout'''
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float):
        assert kernel % 2 == 1
        super().__init__()

        padding = (kernel - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_ch, 
                out_channels=out_ch,
                kernel_size=kernel,
                padding=padding,
            ),
            nn.BatchNorm1d(num_features=out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, tensor: Tensor) -> Tensor: 
        return self.block(tensor)
