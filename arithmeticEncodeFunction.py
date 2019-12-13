import numpy
import torch
from bitstream import BitStream
from collections import namedtuple
from collections import OrderedDict


def valueArithmeticEncode(inputData, numbits):# inputData是一维numpy
    # 获取数据分布
    minV = int(inputData.min())
    maxV = int(inputData.max())
    inputDataHistc = torch.histc(torch.from_numpy(inputData).cuda(), min=minV, max=maxV,
                               bins=(maxV - minV + 1)).cpu().numpy()
    print('inputData的最值，分布分别为', minV, maxV, inputDataHistc)
    # 一共有maxV - minV + 1个值，因此需要把[0,1]分成maxV - minV + 1个区间，区间长度即为概率值
    inputDataP = inputDataHistc / inputDataHistc.sum() # inputDataP[i] 为minV + i的概率
    valueL = numpy.zeros_like(inputDataP)
    valueH = numpy.zeros_like(inputDataP)
    valueH[0] = inputDataP[0]
    for i in range(1, inputDataP.shape[0]):
        valueH[i] = valueH[i-1] + inputDataP[i]
        valueL[i] = valueH[i-1]

    print('初始区间的划分为')
    for i in range(inputDataP.shape[0]):
        print(minV + i, '[', valueL[i], ',', valueH[i], ']')


    L = 0
    H = 1
    thresHold = 0.75
    # 开始算术编码
    for i in range(inputData.shape[0]):
        R = H - L
        H = L + R*valueH[inputData[i] - minV]
        L = L + R*valueL[inputData[i] - minV]
        #assert L!=H, '算术编码器遇到了精度阈值！'
        print(L, H, R)





if __name__ == '__main__':
    



    exit(0)

