import torch
import torch.nn as nn
import torch.nn.functional as F



def createRGGB121Filter(inputData):
    '''
    创建一个用于将RGB444转换成RGB121的滤波器
    :param inputData: batchSize * 3 * m * n
    :return: batchSize * 3 * m * n
    '''
    RGBFilter = torch.zeros(size=[3, 2, 2], device=inputData.device).float()
    RGBFilter[0][0][0] = 1  # R
    RGBFilter[1][0][1] = 1  # G
    RGBFilter[1][1][0] = 1  # G
    RGBFilter[2][1][1] = 1  # B
    return RGBFilter.repeat(1, inputData.shape[2]//2, inputData.shape[3]//2).expand(inputData.shape)

def createHollowFilter(inputData, type=0):
    '''
    :param inputData: 输入数据，多通道，各个维度应该均为偶数
    :param type: type=0则偶数通道是[1 0 奇数通道是[0 1  type=1反之
                                    0 1]           1 0]
    :return: 返回一个可以直接与其相点乘的矩阵
    '''
    filter = torch.zeros(size=[2, 2, 2], device=inputData.device).float()
    filter[type][0][0] = 1
    filter[type][1][1] = 1
    filter[1 - type][0][1] = 1
    filter[1 - type][1][0] = 1
    return filter.repeat(inputData.shape[1] // 2, inputData.shape[2] // 2, inputData.shape[3] // 2).expand(inputData.shape)


def hollowDownSampleTensor(inputData):
    '''
    空洞下采样
    连续4个通道按照如下所示采样
    [1 0  [0 1  [0 0  [0 0
     0 0]  0 0]  1 0]  0 1]


    :param inputData: batchSize * c * w * h
    :param hollowFilter: batchSize * c * w/2 * h/2
    :return:
    '''
    batchSize, c, w, h = inputData.shape
    kernel = torch.zeros(size=[c, 1, 2, 2], device=inputData.device)
    kernel[0::4, 0, 0, 0] = 1
    kernel[1::4, 0, 0, 1] = 1
    kernel[2::4, 0, 1, 0] = 1
    kernel[3::4, 0, 1, 1] = 1
    return F.conv2d(inputData, kernel, stride=2, groups=c)

def hollowUpSampleTensor(inputData):
    '''
    空洞上采样
    连续4个通道按照如下所示上采样
    [1 0  [0 1  [0 0  [0 0
     0 0]  0 0]  1 0]  0 1]

    :param inputData: batchSize * c * w * h
    :param hollowFilter: batchSize * c * 2w * 2h
    :return:
    '''
    batchSize, c, w, h = inputData.shape
    kernel = torch.zeros(size=[c, 1, 2, 2], device=inputData.device)
    kernel[0::4, 0, 0, 0] = 1
    kernel[1::4, 0, 0, 1] = 1
    kernel[2::4, 0, 1, 0] = 1
    kernel[3::4, 0, 1, 1] = 1
    return F.conv_transpose2d(inputData, kernel, stride=2, groups=c)


if __name__ == '__main__':
    x = torch.range(start=0, end=63).reshape([1,4,4,4])
    print(x)
    y = hollowDownSampleTensor(x)
    print(y.shape)
    print(y)
    x_ = hollowUpSampleTensor(y)
    print(x_.shape)
    print(x_)



    exit(0)


