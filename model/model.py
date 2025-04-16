import torch
from torch import nn

#from torchinfo import summary
#from torchvision import models
#from torchsummary import summary

class EventDetector(nn.Module):
    def __init__(self, in_channels, out_channels, classification_head, kernel_size_one=9, kernel_size_all=3, stride_one=1, stride=1, dropout_p=0.1):
        super().__init__()
        layers = []

        layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=kernel_size_one,
                                stride=stride_one, padding='same', padding_mode='zeros'))
        layers.append(nn.GELU())
        layers.append(nn.BatchNorm1d(out_channels[0]))
        #layers.append(nn.MaxPool1d(kernel_size=3))
        layers.append(nn.Dropout(p=dropout_p))
        for i in range(1, len(out_channels)):
            layers.append(
                nn.Conv1d(in_channels=out_channels[i - 1], out_channels=out_channels[i], kernel_size=kernel_size_all,
                          stride=stride, padding='same', padding_mode='zeros'))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(out_channels[i]))
            #layers.append(nn.MaxPool1d(kernel_size=5))
            layers.append(nn.Dropout(p=dropout_p))

        self.module_list = nn.ModuleList(layers)
        #self.classification_head = nn.ModuleList([nn.Linear(out_channels[-1], 256), nn.GELU(), nn.Linear(256, 32), nn.GELU(), nn.Linear(32, 1)])
        #self.classification_head = nn.Sequential(nn.Linear(out_channels[-1], 256), nn.GELU(), nn.Linear(256, 32), nn.GELU(), nn.Linear(32, 1))
        self.classification_head = nn.Sequential(nn.Linear(classification_head[0], classification_head[1]))
        for i in range(1, len(classification_head) - 1):
            appended = self.classification_head.append(nn.GELU())
            appended = self.classification_head.append(nn.Linear(classification_head[i], classification_head[i+1]))
        #self.classification_head = nn.ModuleList([nn.Linear(out_channels[-1], 64), nn.Linear(64, 1)])
        #self.classification_head = nn.ModuleList([nn.Linear(out_channels[-1], 1)])
        #self.classification_head = nn.Linear(out_channels[-1], 1)
        
    def forward(self, x):
        for l in self.module_list:
            x = l.forward(x)
        x = torch.swapaxes(x, 1, 2)
        for l in self.classification_head:
            x = l(x)
        #x = self.classification_head(x)
        return x


#model = EventDetector(in_channels=1, out_channels=[32, 64, 128, 256, 512, 1024, 2048, 1024], kernel_size_one=3, kernel_size_all=9)
#print(summary(model, (1, 6000)))
