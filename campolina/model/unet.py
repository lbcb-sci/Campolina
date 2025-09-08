import torch
from torch import nn, Tensor
from torch.utils import checkpoint

# constants
INPUT_CHANNELS = 4 # n features
KERNEL = 3 # kernel size of encoder and decoder layers
KERNEL_FIRST = 9 # size of kernel of first layer
KERNEL_BOTTLENECK = 5 # size of kernel at bottleneck

class UNet(nn.Module):
    '''
    (Currently being tested, see `Default` for the official Campolina model.)
    '''

    def __init__(
            self,
            channels: list[int],
            checkpoint: bool = False,
        ):

        super().__init__()

        rchannels = list(reversed(channels))

        self.first = ConvBlock(
            INPUT_CHANNELS, 
            channels[0], 
            KERNEL_FIRST,
        )

        self.downlayers = nn.ModuleList([
            Down(in_ch=in_ch, out_ch=out_ch, kernel=KERNEL)
            for (in_ch, out_ch) in zip(channels[:-1], channels[1:])
        ])

        self.middle = ConvBlock(channels[-1], channels[-1], KERNEL_BOTTLENECK)

        self.uplayers = nn.ModuleList([
            Up(in_ch=in_ch*2, out_ch=out_ch, kernel=KERNEL, checkpoint=checkpoint)
            for (in_ch, out_ch) in zip(rchannels[:-1], rchannels[1:])
        ])

        self.last = nn.Conv1d(channels[0]*2, 1, kernel_size=1)

    @classmethod
    def make_default(cls, checkpoint = False):
        return cls(
            channels=[64, 128, 128, 128, 256], 
            checkpoint=checkpoint,
        )

    def downsample(self, tensor: Tensor) -> tuple[Tensor, list]:
        '''Apply the layers in `self.downlayers` and track residuals.'''

        residuals = []
        for down in self.downlayers:
            tensor = down(tensor)
            residual = tensor
            residuals.append(residual)

        return tensor, residuals

    def upsample(self, tensor: Tensor, residuals: list[Tensor]) -> Tensor:
        '''Apply the layers in `self.uplayers` with concatenated residuals.'''

        for i, up in enumerate(self.uplayers):
            residual = residuals[-(i+1)]
            tensor = torch.concat([tensor, residual], dim=1)
            tensor = up(tensor)

        return tensor

    def forward(self, tensor: Tensor) -> Tensor:
        # first layer 
        first_out = self.first(tensor)

        # downsample
        downsample_out, residuals = self.downsample(first_out)

        # bottleneck
        middle_out = self.middle(downsample_out)

        # upsample
        upsample_out = self.upsample(middle_out, residuals)

        # last
        input_last = torch.concat([upsample_out, first_out], dim=1)
        final_out = self.last(input_last).squeeze(1)

        return final_out

class Down(nn.Module):
    '''Halves the input by applying DoubleConv + Pooling.'''

    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        self.layers = nn.Sequential(DoubleConvBlock(in_ch, out_ch, kernel), nn.MaxPool1d(2, 2))

    def forward(self, tensor: Tensor) -> Tensor: return self.layers(tensor)

class Up(nn.Module):
    '''Upsample the input using Nearest Upsampling + DoubleConv.'''

    def __init__(self, in_ch: int, out_ch: int, kernel: int, checkpoint: bool):
        super().__init__()
        assert kernel % 2 == 1
        self.ckpt = checkpoint
        self.layers = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), DoubleConvBlock(in_ch, out_ch, kernel))

    def forward(self, tensor: Tensor) -> Tensor:
        if not self.ckpt: return self.layers(tensor)
        return checkpoint.checkpoint(lambda _: self.layers(_), tensor, use_reentrant=False)

class DoubleConvBlock(nn.Module):
    '''Apply two 1d convolutions in a row, with mid_channels = max(in_channels, out_channels).'''

    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        mid_ch = max(in_ch, out_ch)
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)
        self.layers = nn.Sequential(ConvBlock(in_ch, mid_ch, kernel), ConvBlock(mid_ch, out_ch, kernel))

    def forward(self, tensor: Tensor) -> Tensor: return self.layers(tensor) + self.proj(tensor)

class ConvBlock(nn.Module):
    '''Convolution + BatchNorm + GELU'''

    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        assert kernel % 2 == 1

        super().__init__()
        padding = (kernel - 1) // 2 # pad to same size

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_ch, 
                out_channels=out_ch,
                kernel_size=kernel,
                padding=padding,
            ),
            nn.BatchNorm1d(num_features=out_ch),
            nn.GELU(),
        )

    def forward(self, tensor: Tensor) -> Tensor: return self.block(tensor)
