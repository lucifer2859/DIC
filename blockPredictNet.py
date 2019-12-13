import torch
import torch.nn as nn
import torch.nn.functional as F
import baseNet
class blockPredictNet(nn.Module):
    '''
    输入是3*N*N
    输出为N*N。用左上、左、上的区域来预测右下区域的像素值。
    '''
    def __init__(self, channels, depth):
        super(blockPredictNet, self).__init__()
        self.conv_channels_up = nn.Conv2d(3, channels, 1)
        convList = []
        for i in range(depth):
            convList.append(baseNet.ResNet(transpose=False, channels=channels, kernel_size=3, padding=1))
        self.conv_channels_down = nn.Conv2d(channels, 1, 1)
        self.convList = nn.Sequential(*convList)
    def forward(self, x):
        return F.leaky_relu_(self.conv_channels_down(self.convList(F.leaky_relu_(self.conv_channels_up(x)))))