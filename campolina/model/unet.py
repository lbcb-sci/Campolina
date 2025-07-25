import torch
from torch import nn, Tensor

class UNet(nn.Module):
    def __init__(
            self,
            chan_down: list[int],
            chan_up: list[int],
            kernel: int = 3,
            dropout: float = 0.1,
        ):
        assert len(chan_down) == len(chan_up)
        super().__init__()
        self.name= 'UNet'

        self.first = ConvBlock(
            in_ch=5,
            out_ch=chan_down[0],
            kernel=kernel,
            dropout=0.0,
        )

        self.downlayers = nn.ModuleList([
            Down(
                in_ch=chan_down[i], 
                out_ch=chan_down[i+1],
                kernel=kernel,
                dropout=dropout,
            ) for i, _ in enumerate(chan_down[:-1])
        ])

        self.uplayers = []
        for i, _ in enumerate(chan_up[:-1]):
            self.uplayers.append(Up(
                in_ch=chan_up[i] + chan_down[-i-1], # residual
                out_ch=chan_up[i+1],
                kernel=kernel,
                dropout=dropout,
            ))
        self.uplayers = nn.ModuleList(self.uplayers)

        self.last = nn.Conv1d(
            in_channels=chan_down[0]+chan_up[-1], # first + output of upsample
            out_channels=1,
            kernel_size=3,
            padding=1,
        )

    @staticmethod
    def make_default(dropout: float):
        chan = [5, 64, 128, 256, 512]
        return UNet(chan_down=chan, chan_up=list(reversed(chan)), dropout=dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.first(x)
        first = x.clone()

        residuals = []

        # down sample
        for down in self.downlayers:
            x = down(x)
            residuals.append(x.clone())

        # up sample
        for i, up in enumerate(self.uplayers):
            residual = residuals[-i-1]
            x = up(torch.concat([x, residual], dim=1))
        
        # get logits
        return self.last(torch.concat([x, first], dim=1))

class Down(nn.Module):
    '''Halves the input by applying pooling + convolution.'''
    def __init__(
            self, 
            in_ch: int, 
            out_ch: int,
            kernel: int,
            pooling: int = 2,
            dropout: float = 0.1,
        ):

        super().__init__()

        self.maxpool = nn.MaxPool1d(pooling)

        self.conv = nn.Sequential(
            ConvBlock(
                in_ch=in_ch, 
                out_ch=out_ch, 
                kernel=kernel, 
                dropout=dropout,
            ),
            #Convolution(
                #in_ch=out_ch, 
                #out_ch=out_ch, 
                #kernel=kernel, 
                #dropout=dropout,
            #),
        )

    def forward(self, x: Tensor) -> Tensor: return self.conv(self.maxpool(x))

class Up(nn.Module):
    '''Upsample the input using transposed_conv + conv.'''
    def __init__(
            self, 
            in_ch: int, 
            out_ch: int,
            kernel: int,
            dropout: float = 0.1,
        ):

        super().__init__()

        assert kernel % 2 == 1

        # TODO try other upsampling strategies
        self.convT = nn.ConvTranspose1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=2,
            stride=2,
        )

        self.conv = nn.Sequential(
            ConvBlock(
                in_ch=out_ch, 
                out_ch=out_ch, 
                kernel=kernel, 
                dropout=dropout,
            ),
            #Convolution(
                #in_ch=out_ch, 
                #out_ch=out_ch, 
                #kernel=kernel, 
                #dropout=dropout,
            #),
        )

    def forward(self, x: Tensor) -> Tensor: return self.conv(self.convT(x))

class ConvBlock(nn.Module):
    '''Conv + Norm + ReLU + Dropout'''
    def __init__(
            self, 
            in_ch: int, 
            out_ch: int,
            kernel: int,
            dropout: float = 0.1,
        ):
        assert kernel % 2 == 1
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_ch, 
                out_channels=out_ch,
                kernel_size=kernel,
                padding=(kernel - 1) // 2,
            ),
            nn.GroupNorm(
                num_groups=min(out_ch, 32),
                num_channels=out_ch,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor: return self.block(x)

    