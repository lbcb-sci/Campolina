import torch
from torch import nn, Tensor

class UNet(nn.Module):
    def __init__(
            self,
            channels_down: list[int],
            channels_up: list[int],
            kernels: list[int] = None,
            dropout: float = 0.1,
        ):

        assert len(channels_down) == len(channels_up)

        if kernels is None: kernels = [3] * (len(channels_down * 2))

        assert len(channels_down)*2 == len(kernels)

        super().__init__()
        self.name= 'UNet'

        self.downlayers = []
        for i, _ in enumerate(channels_down[:-1]):
            self.downlayers.append(Down(
                in_ch=channels_down[i],
                out_ch=channels_down[i+1],
                kernel=kernels[i],
                dropout=dropout,
            ))

        self.uplayers = []
        for i, _ in enumerate(channels_up[:-1]):
            self.uplayers.append(Up(
                in_ch=channels_up[i] + channels_down[-i-1], # residual
                out_ch=channels_up[i+1],
                kernel=kernels[len(channels_down) + i],
                dropout=dropout,
            ))

        self.first = Convolution(
            in_ch=5,
            out_ch=channels_down[0],
            kernel=3,
            dropout=0.0,
        )

        self.downlayers = nn.ModuleList(self.downlayers)

        #self.middle = Convolution(
            #in_ch=channels_down[-1],
            #out_ch=channels_down[-1],
            #kernel=3,
            #dropout=dropout,
        #)

        self.uplayers = nn.ModuleList(self.uplayers)

        self.last = nn.Conv1d(
            in_channels=channels_down[0]+channels_up[-1],
            out_channels=1,
            kernel_size=3,
            padding=1,
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.first(x)
        first = x.clone()

        residuals = []

        # down sample
        for down in self.downlayers:
            x = down(x)
            residuals.append(x.clone())

        #x = self.middle(x)

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

        self.conv = nn.Sequential(
            nn.MaxPool1d(pooling),

            Convolution(
                in_ch=in_ch, 
                out_ch=out_ch, 
                kernel=kernel, 
                dropout=dropout,
            ),

            Convolution(
                in_ch=out_ch, 
                out_ch=out_ch, 
                kernel=kernel, 
                dropout=dropout,
            ),
        )

    def forward(self, x: Tensor) -> Tensor: return self.conv(x)

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
        self.transposed_conv = nn.ConvTranspose1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=2,
            stride=2,
        )

        self.conv = nn.Sequential(
            Convolution(
                in_ch=out_ch, 
                out_ch=out_ch, 
                kernel=kernel, 
                dropout=dropout,
            ),
            Convolution(
                in_ch=out_ch, 
                out_ch=out_ch, 
                kernel=kernel, 
                dropout=dropout,
            ),
        )

    def forward(self, x: Tensor) -> Tensor: return self.conv(self.transposed_conv(x))

class Convolution(nn.Module):
    '''Conv + BatchNorm + ReLU'''
    def __init__(
            self, 
            in_ch: int, 
            out_ch: int,
            kernel: int,
            dropout: float = 0.1,
        ):
        assert kernel % 2 == 1
        super().__init__()

        self.conv_norm = nn.Sequential(
            nn.Conv1d(
                in_channels=in_ch, 
                out_channels=out_ch,
                kernel_size=kernel,
                padding=(kernel - 1) // 2,
            ),
            nn.BatchNorm1d(num_features=out_ch),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor: return self.conv_norm(x)

def make_small(dropout: float = 0.1) -> UNet:
    # ~2.7M params
    channels = [5, 32, 64, 128, 256]
    unet = UNet(
        channels_down=channels,
        channels_up=list(reversed(channels)),
        dropout=dropout,
    )
    unet.name = 'unet_small'
    return unet

def make_medium(dropout: float = 0.1) -> UNet:
    # ~7M Params
    channels = [5, 64, 128, 256, 512]
    unet = UNet(
        channels_down=channels,
        channels_up=list(reversed(channels)),
        dropout=dropout,
    )
    unet.name = 'unet_medium'
    return unet

def make_big(dropout: float = 0.1) -> UNet:
    channels = [64, 128, 256, 512, 512]
    unet = UNet(
        channels_down=channels,
        channels_up=list(reversed(channels)),
        dropout=dropout,
    )
    unet.name = 'unet_big'
    return unet