import torch
from torch import nn, Tensor
from torch.nn import functional as F

class UNet(nn.Module):
    def __init__(
            self, 
            kernels: list[int],
            dropout: float = 0.1,
        ):
        super().__init__()

        self.name= 'UNet'

        self.first = Convolution(
            in_ch=5,
            out_ch=16,
            kernel=kernels[0],
            dropout=dropout,
        )

        self.down1 = Down(
            in_ch=16, 
            out_ch=32, 
            kernel=kernels[1],
            dropout=dropout,
        )

        self.down2 = Down(
            in_ch=32, 
            out_ch=64, 
            kernel=kernels[2],
            dropout=dropout,
        )

        self.down3 = Down(
            in_ch=64, 
            out_ch=64, 
            kernel=kernels[3],
            dropout=dropout,
        )

        self.down4 = Down(
            in_ch=64, 
            out_ch=64, 
            kernel=kernels[4],
            dropout=dropout,
        )

        self.middle = Convolution(
            in_ch=64, 
            out_ch=64, 
            kernel=kernels[5],
            dropout=dropout,
        )

        self.up4 = Up(
            in_ch=64, 
            out_ch=64, 
            kernel=kernels[6],
            dropout=dropout,
        )

        self.up1 = Up(
            in_ch=64 * 2, 
            out_ch=64, 
            kernel=kernels[7],
            dropout=dropout,
        )

        self.up2 = Up(
            in_ch=64 * 2, 
            out_ch=32, 
            kernel=kernels[8],
            dropout=dropout,
        )

        self.up3 = Up(
            in_ch=32 * 2, 
            out_ch=8, 
            kernel=kernels[9],
            dropout=dropout,
        )

        #self.last = nn.Linear(in_features=, out_features=1)

        self.last = nn.Conv1d(
            in_channels=16 + 8, 
            out_channels=1, 
            kernel_size=1,#kernels[10],
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x : 5 x 6000

        first = self.first(x)
        # first : 8 x 6000

        d1 = self.down1(first)
        # d1 : 16 x 3000

        d2 = self.down2(d1)
        # d2 : 32 x 1500

        d3 = self.down3(d2)
        # d3 : 64 x 750

        d4 = self.down4(d3)
        # d3 : 128 x 375

        mid = self.middle(d4)
        ## mid: 128 x 375

        u4 = self.up4(mid)# + d3
        # u4 :  64 x 375
        u4 = torch.concat([u4, d3], dim=1)
        # u4 :  128 x 375

        u1 = self.up1(u4)# + d2
        # u1: 32 x 1500
        u1 = torch.concat([u1, d2], dim=1)
        # u1: 64 x 1500

        u2 = self.up2(u1) #+ d1
        # u2: 16 x 3000
        u2 = torch.concat([u2, d1], dim=1)
        # u2: 32 x 3000

        u3 = self.up3(u2)# + first
        # u1: 8 x 6000
        u3 = torch.concat([u3, first], dim=1)
        # u3: 16 x 3000

        return self.last(u3)

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

        #self.conv = Convolution(
            #in_ch=in_ch, 
            #out_ch=out_ch, 
            #kernel=kernel, 
            #dropout=dropout,
        #)

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
        )

        self.pooling = pooling

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool1d(self.conv(x), self.pooling)

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
            out_channels=in_ch,
            kernel_size=2,
            stride=2,
        )

        #self.conv = Convolution(
            #in_ch=in_ch, 
            #out_ch=out_ch, 
            #kernel=kernel,
            #dropout=dropout,
        #)

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
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.transposed_conv(x)
        x = self.conv(x)
        return x

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

        self.dropout = dropout

        self.conv_norm = nn.Sequential(
            nn.Conv1d(
                in_channels=in_ch, 
                out_channels=out_ch,
                kernel_size=kernel,
                padding=(kernel - 1) // 2,
            ),

            nn.BatchNorm1d(num_features=out_ch)
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(F.relu(self.conv_norm(x)), self.dropout)
