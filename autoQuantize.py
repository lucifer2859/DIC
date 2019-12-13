import torch
import torch.nn.functional as nnF
from matplotlib import pyplot
import numpy

'''
自动量化
inputData: 输入的一维数据，有序
qLevel: 允许使用的量化值的数量
要求求解出最优的量化取值集合C={c_0,c_1,...,c_(qLevel-1)}

假设我们找到了这样的最优集合C
量化函数记为f，对于inputData[i]，f(inputData[i])=c_j
error = ∑(inputData[i] - f(inputData[i]))^2

相当于要将inputData的数据分成qLevel段，每段内的数值用一个数代替
相当于在[inputData.min(), inputData.max()]区间上，找(qLevel-1)个点，将区间分成qLevel段
每一段内的数据，在量化后的取值为这一段数据的均值（一段数据与它们的均值的平方距离和最小）
记这些用来分割的点为S={x_0, x_1, ..., x_(qLevel-2)}
最开始时，将区间均匀划分，得到初始的S_0
不难看出，对于x_i的左右移动，位于[inputData.min(), x_(i-1))和[x_(i+1), inputData.max())的量化误差不会改变
问题退化为在[x_(i-1), x_(i+1)]内找到x_i使得左右两个区间的量化误差最小
'''

def autoQuantize(inputData, qLevel, printFlag = False):
    '''
    :param inputData: 有序一维Tensor
    :param qLevel: 量化集合中的值的数量
    :return: numpy数组
    '''
    segmentArray = numpy.zeros(shape=[qLevel+1], dtype=int)
    segmentArray[0] = 0
    segmentArray[qLevel] = inputData.shape[0] + 1 # 由于采取左包含右不包含的[ , )区间形式，因此最右端+1

    # 首先在[L,R]中找qLevel-1个分割点，将区间分成qLevel段
    tempLength = segmentArray[qLevel] // qLevel
    for i in range(1, qLevel-1):
        segmentArray[i] = i * tempLength

    segmentArray[qLevel - 1] = (segmentArray[qLevel - 2] + segmentArray[qLevel])//2
    if(printFlag==True):
        print('初始的划分为', segmentArray)
    # 开始迭代划分
    iterNum = 0
    lastSumError = 0
    while (True):
        for i in range(1, qLevel):
            segmentArray[i] = searchBetterSegment(inputData, segmentArray[i - 1], segmentArray[i + 1])
        sumError = 0
        for i in range(qLevel):
            sumError = sumError + torch.var(inputData[ segmentArray[i]:segmentArray[i+1] ], unbiased=False).item() * (segmentArray[i+1] - segmentArray[i])
        iterNum = iterNum + 1
        if(printFlag==True):
            print('第', iterNum, '次迭代后得到的总误差为', sumError)
        if(lastSumError==sumError):
            break
        lastSumError = sumError

    ret = numpy.zeros(shape=qLevel, dtype=float)
    for i in range(qLevel):
        ret[i] = torch.mean(inputData[ segmentArray[i]:segmentArray[i+1] ]).item()
    return ret
def searchBetterSegment(inputData, L, R):
    '''
    :param inputData: 有序一维Tensor
    :param L: inputData[L]是进行操作的区间的左端
    :param R: inputData[R]是进行操作的区间的右端
    :return: M
    '''
    minError = float("inf")
    M = 0
    for i in range(L+1, R):

        errorL = torch.var(inputData[L:i], unbiased=False) * (i - L) # [L, i) = [L, i-1]
        errorR = torch.var(inputData[i:R], unbiased=False) * (R - i) # [i, R) = [i, R-1]
        error = errorL.item() + errorR.item()
        if(error < minError):
            minError = error
            M = i
    return M

def nearestQuantize(inputData, C):
    '''
    :param inputData: 任意形状的Tensor
    :param C: shape=[qLevel, 1]的tensor，量化后的取值集合
    :return: 量化后的结果
    '''
    x = inputData.view(-1).repeat(C.shape[0], 1)
    dTensor = torch.abs(x - C)
    # dTensor[i]是与C[i]的距离
    return C[dTensor.argmin(0)].t().reshape(inputData.shape)

if __name__ == '__main__':
    inputData = torch.rand(size=[1000]).sort().values
    print(autoQuantize(inputData, 16))







