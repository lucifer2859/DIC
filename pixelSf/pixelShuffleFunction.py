from PIL import Image
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim

def pixel_unshuffle(input, downscale_factor):
    '''
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    相当于把输入的每一个通道中的k*w * k*h，拆分成k个w * h
    用kernel = [k*k*c, 1, k, k]的卷积核进行groups=c的卷积即可
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        '''
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        相当于把输入的每一个通道中的k*w * k*h，拆分成k个w * h
        用kernel = [k*k*c, 1, k, k]的卷积核进行groups=c的卷积即可
        '''
        c = input.shape[1]

        kernel = torch.zeros(size=[self.downscale_factor*self.downscale_factor*c,
                                   1, self.downscale_factor, self.downscale_factor],
                             device=input.device)

        for y in range(self.downscale_factor):
            for x in range(self.downscale_factor):
                kernel[x + y * self.downscale_factor::self.downscale_factor*self.downscale_factor, 0, y, x] = 1
        return F.conv2d(input, kernel, stride=self.downscale_factor, groups=c)




if __name__ == '__main__':
    x = torch.range(start=0, end=127).reshape([1,8,4,4])
    print(x)
    y = F.pixel_shuffle(x, 2)
    print(y.shape)
    print(y)
    x_ = pixel_unshuffle(y, 2)
    print(x_.shape)
    print(x_)
    exit(0)

