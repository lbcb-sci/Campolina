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

        self.down = []
        for i, _ in enumerate(channels_down[:-1]):
            self.down.append(Down(
                in_ch=channels_down[i],
                out_ch=channels_down[i+1],
                kernel=kernels[i],
                dropout=dropout,
            ))

        self.up = []
        for i, _ in enumerate(channels_up[:-1]):
            self.up.append(Up(
                in_ch=channels_up[i] + channels_down[-i-1], # residual
                out_ch=channels_up[i+1],
                kernel=kernels[len(channels_down) + i],
                dropout=dropout,
            ))

        self.down = nn.ModuleList(self.down)

        #self.middle = Convolution(
            #in_ch=channels_down[-1],
            #out_ch=channels_down[-1],
            #kernel=3,
            #dropout=dropout,
        #)

        self.up = nn.ModuleList(self.up)

        self.last = nn.Conv1d(
            in_channels=10,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )
        
    def forward(self, x: Tensor) -> Tensor:
        inp = x.clone()

        residuals = []

        # down sample
        for down in self.down:
            x = down(x)
            residuals.append(x.clone())

        #x = self.middle(x)

        # up sample
        for i, up in enumerate(self.up):
            residual = residuals[-i-1]
            x = up(torch.concat([x, residual], dim=1))
        
        # get logits
        cat = torch.concat([x, inp], dim=1)
        return self.last(cat)

class Down(nn.Module):
    '''Halves the input by applying convolution + max pooling.'''
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
            nn.MaxPool1d(pooling)
        )

    def forward(self, x: Tensor) -> Tensor: return self.conv(x)

class Up(nn.Module):
    '''Upsample the input using transposed conv + conv.'''
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

        super().__init__()

        assert kernel % 2 == 1

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
    # ~11M params
    channels = [5, 128, 256, 512, 1024]
    unet = UNet(
        channels_down=channels,
        channels_up=list(reversed(channels)),
        dropout=dropout,
    )
    unet.name = 'unet_big'
    return unet