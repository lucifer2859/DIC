import torch
import torch.nn.functional as F
import numpy

def informationEntropy(inputData, C, sigma=3, mean=True):
    '''
    :param inputData: 量化后的数据，batchSize * ?
    :param C: 量化取值集合，L行1列
    :param sigma: 越大，算出的频率越接近实际频率，但导数会越趋近0，容易出现梯度消失
    近似脉冲函数为exp(-sigma*x^2)
    :param mean: 为True则返回batchSize个数据的平均，否则返回一个size=[batchSize]的Tensor
    :return: 信息熵
    '''
    IE = torch.zeros(size=[inputData.shape[0]], device=inputData.device)
    for i in range(inputData.shape[0]):
        approximateP = torch.zeros_like(C).float()
        for j in range(C.shape[0]):
            approximateP[j] = torch.exp(-sigma * torch.pow(inputData[i] - C[j], 2)).sum() + 1e-6
            # + 1e-6防止出现log2(0)
        approximateP = approximateP / approximateP.sum()
        IE[i] = -torch.sum(approximateP * torch.log2(approximateP))
    if(mean==True):
        return IE.mean() # 返回batchSize个数据的平均信息熵
    else:
        return IE

def fluctuationEntropy(inputData, sigma=3, mean=True):
    '''
    :param inputData: 量化后的数据，batchSize * ?
    :param sigma: 使用exp(-sigma*x^2)对卷积后的数据进行处理
    sigma越大，非0元素处理后越接近1，但导数会越趋近0，容易出现梯度消失
    :param mean: 为True则返回batchSize个数据的平均，否则返回一个size=[batchSize]的Tensor
    :return: 波动熵
    '''
    kernel = torch.tensor([1, -1], device=inputData.device)\
        .float().reshape([1, 1, 2]).repeat(inputData.shape[0], 1, 1)
    flatInputData = inputData.view(inputData.shape[0], 1, -1)
    fluctuationData = 1 - torch.exp(-sigma * torch.pow(F.conv1d(flatInputData, kernel), 2))
    if(mean==True):
        return fluctuationData.mean()
    else:
        ret = torch.zeros(size=[flatInputData.shape[0]], device=flatInputData.device)
        for i in range(flatInputData.shape[0]):
            ret[i] = fluctuationData[i].mean()
        return ret

if __name__ == '__main__':
    exit(0)