import torch.nn as nn
import torch.nn.functional as F
import pytorch_gdn

class SampleNet(nn.Module): # 默认核为2，步长为2的升/降采样网络，通过卷积/反卷积实现，包含GDN/IGDN层
    def __init__(self, downSample, in_channels, out_channels, kernel_size=2, stride=2, padding=0, groups=1, step=2):
        super(SampleNet, self).__init__()
        self.downSample = downSample
        self.step = step
        middle_channels = (in_channels + out_channels)//2
        if(self.downSample==True):
            if(self.step==2):
                self.convDownX = nn.Conv2d(in_channels, middle_channels, (kernel_size, 1), (stride, 1), groups=groups)

                self.convDownY = nn.Conv2d(middle_channels, out_channels, (1, kernel_size), (1, stride), groups=groups)
            elif(self.step==1):
                self.convDown = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups)

            self.gdn = pytorch_gdn.GDN(out_channels)

        elif(self.downSample==False):

            self.igdn = pytorch_gdn.GDN(in_channels, True)
            if (self.step == 2):
                self.tconvUpY = nn.ConvTranspose2d(in_channels, middle_channels, (1, kernel_size), (1, stride), groups=groups)
                self.tconvUpX = nn.ConvTranspose2d(middle_channels, out_channels, (kernel_size, 1), (stride, 1), groups=groups)
            elif (self.step == 1):
                self.tconvUp = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, groups=groups)


    def forward(self, x):
        if(self.downSample==True):
            if(self.step==2):
                return self.gdn(F.leaky_relu_(self.convDownY(F.leaky_relu_(self.convDownX(x)))))
            elif(self.step==1):
                return self.gdn(F.leaky_relu_(self.convDown(x)))
        elif(self.downSample==False):
            if(self.step == 2):
                return F.leaky_relu_(self.tconvUpX(F.leaky_relu_(self.tconvUpY(self.igdn(x)))))
            elif(self.step==1):
                return F.leaky_relu_(self.tconvUp(self.igdn(x)))


class ResNet(nn.Module): # 2层残差网络
    def __init__(self, transpose, channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', batch_norm=True, skip=True):
        super(ResNet, self).__init__()
        self.batch_norm = batch_norm
        self.skip = skip
        if (transpose==False):
            self.conv1X = nn.Conv2d(channels, channels, (kernel_size, 1), (stride, 1),
                                    (padding, 0), dilation, groups,
                                   bias, padding_mode)
            self.conv1Y = nn.Conv2d(channels, channels, (1, kernel_size), (1, stride),
                                    (0, padding), dilation, groups,
                                    bias, padding_mode)
            if (self.batch_norm==True):
                self.bn1 = nn.BatchNorm2d(channels)
                self.bn2 = nn.BatchNorm2d(channels)

            self.conv2X = nn.Conv2d(channels, channels, (kernel_size, 1), (stride, 1),
                                    (padding, 0), dilation, groups, bias, padding_mode)
            self.conv2Y = nn.Conv2d(channels, channels, (1, kernel_size), (1, stride),
                                    (0, padding), dilation, groups, bias, padding_mode)

        elif (transpose==True):
            self.conv1X = nn.ConvTranspose2d(channels, channels, (kernel_size, 1), (stride, 1),
                                             (padding, 0), output_padding=0, groups=groups, bias=bias,
                                             dilation=dilation, padding_mode=padding_mode)
            self.conv1Y = nn.ConvTranspose2d(channels, channels, (1, kernel_size), (1, stride),
                                    (0, padding), output_padding=0, groups=groups, bias=bias,
                                             dilation=dilation, padding_mode=padding_mode)
            if (self.batch_norm == True):
                self.bn1 = nn.BatchNorm2d(channels)
                self.bn2 = nn.BatchNorm2d(channels)

            self.conv2X = nn.ConvTranspose2d(channels, channels, (kernel_size, 1), (stride, 1),
                                             (padding, 0), output_padding=0, groups=groups, bias=bias,
                                             dilation=dilation, padding_mode=padding_mode)
            self.conv2Y = nn.ConvTranspose2d(channels, channels, (1, kernel_size), (1, stride),
                                             (0, padding), output_padding=0, groups=groups, bias=bias,
                                             dilation=dilation, padding_mode=padding_mode)

    def forward(self, x):
        if (self.batch_norm == True):
            if (self.skip == True):
                return F.leaky_relu_(x + self.bn2(self.conv2Y(self.conv2X(F.leaky_relu_(self.bn1(self.conv1Y(self.conv1X(x))))))))
            return F.leaky_relu_(self.bn2(self.conv2Y(self.conv2X(F.leaky_relu_(self.bn1(self.conv1Y(self.conv1X(x))))))))
        else:
            if (self.skip == True):
                return F.leaky_relu_(x + self.conv2Y(self.conv2X(F.leaky_relu_(self.conv1Y(self.conv1X(x))))))
            return F.leaky_relu_(self.conv2Y(self.conv2X(F.leaky_relu_(self.conv1Y(self.conv1X(x))))))


def setModelParamRequiresGrad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad