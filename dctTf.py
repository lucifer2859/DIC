import torch
import numpy as np
def createDCTKernel(N=8):
    kernel = np.zeros(shape=[N, N], dtype=float)
    for i in range(0, N):
        for j in range(N):
            if(i==0):
                kernel[i][j] = np.sqrt(1 / N) * np.cos((j + 0.5) * np.pi * i / N)
            else:
                kernel[i][j] = np.sqrt(2 / N) * np.cos((j + 0.5) * np.pi * i / N)
    return kernel

def tensorDCT(inputData, DCTKernel, inverse=False):
    '''
    :param inputData: m * m
    :param DCTKernel: 8*8 or 16*16 or bigger
    '''
    N = DCTKernel.shape[0]
    DCTKernelT = torch.t(DCTKernel)
    for y in range(0, inputData.shape[0], N):
        for x in range(0, inputData.shape[1], N):
            if (inverse == True):
                inputData[y:y + N, x:x + N] = torch.mm(DCTKernelT, inputData[y:y + N, x:x + N]).mm(DCTKernel)
            else:
                inputData[y:y + N, x:x + N] = torch.mm(DCTKernel, inputData[y:y + N, x:x + N]).mm(DCTKernelT)


def splitDCTTensor(inputData, DCTKernelSize=8):
    '''
    记DCTKernelSize=N
    inputData: 经过DCT变换的数据，尺寸是m*m，共有m/N个N*N小块
    每个小块中有N*N中频率
    将每个小块中的N*N种频率都抽取出，相同频率的放在同一个通道内，因此总通道数应该为N*N
    因此inputData: m*m -> ret: (N*N) * m/N * m/N 注意后者是一个3维矩阵
    N*N小块中(i,j)位置对应ret[i*N + j]
    ret[k]的原始频率为(k//N, k%N)
    '''
    ret = torch.zeros(size=[DCTKernelSize*DCTKernelSize,
                            inputData.shape[0]//DCTKernelSize, inputData.shape[1]//DCTKernelSize], device=inputData.device)
    for i in range(DCTKernelSize):
        for j in range(DCTKernelSize):
            ret[i*DCTKernelSize + j] = inputData[i::DCTKernelSize, j::DCTKernelSize]
    return ret

def mergeDCTTensor(inputData, DCTKernelSize=8):
    '''
    splitDCTTensor(inputData, DCTKernelSize=8)的逆变换
    inputData: (N*N) * m/N * m/N -> ret: m*m
    '''
    m = inputData.shape[1] * DCTKernelSize
    ret = torch.zeros(size=[m, m], device=inputData.device)
    for i in range(DCTKernelSize):
        for j in range(DCTKernelSize):
            ret[i::DCTKernelSize, j::DCTKernelSize] = inputData[i*DCTKernelSize + j]
    return ret

def createFrequencyFilter(DCTKernelSize, device):
    '''
    返回一个list，list[i]与DCT变换后的输入数据点乘，频率不为i的全部会被剔除
    '''
    ret = []
    for i in range(2*DCTKernelSize - 1):
        kernel = torch.zeros(size=[DCTKernelSize, DCTKernelSize], device=device).float()
        for x in range(DCTKernelSize):
            for y in range(DCTKernelSize):
                if(x+y==i):
                    kernel[x][y] = 1
        ret.append(kernel)
    return ret
def createLRFrequencyFilter(DCTKernelSize, device, L, R):
    '''
    返回一个频率在行列和属于[L,R)则元素为1，其余元素均为0的矩阵
    '''
    kernel = torch.zeros(size=[DCTKernelSize, DCTKernelSize], device=device).float()
    for x in range(DCTKernelSize):
        for y in range(DCTKernelSize):
            if(x+y>=L and x+y<R):
                kernel[x][y] = 1
    return kernel

def splitImageByFrequency(inputData, frequencyFilter, DCTKernel):
    '''
    根据频率，将图片分离成多张图片
    :param inputData: 图片（不是DCT变换后的数据），m * m
    :param frequencyFilter: createFrequencyFilter(DCTKernelSize, device)的返回值
    :param DCTKernel: m * m
    :return: 一个list，list[i]是频率为i的图片，尺寸也是m*m
    不同频率的图片是可以线性叠加的，例如如果将所有list[i]进行求和，就可以得到原图片inputData
    注意，这个函数会将inputData也进行DCT变换
    '''
    tensorDCT(inputData, DCTKernel)
    ret = []
    for i in range(frequencyFilter.__len__()):
        ret.append(inputData * frequencyFilter[i])
        tensorDCT(ret[i], DCTKernel, True)
    return ret

def getImgFrequencyComponent(inputData, LRFilter, DCTKernel):
    '''
    LRFilter与inputData均为m*m
    LRFilter中除了行列之和在[L,R)的位置元素为1，其余均为0
    返回一张图片的频率在[L, R)的分量
    '''
    return DCTKernel.t().mm(DCTKernel.mm(inputData).mm(DCTKernel.t()) * LRFilter).mm(DCTKernel)







def splitImageByFrequencyToList(inputData, DCTKernel):
    '''
    将输入的图片先进行DCT变换，再将不同频率保存到list中，list[i]保存频率为i的数据
    按光栅顺序扫描DCT变换后的矩阵
    DCTKernel应该与inputData均为m*m的尺寸
    '''
    tensorDCT(inputData, DCTKernel)
    ret = []
    for i in range(2*inputData.shape[0] - 1):
        ret.append([])
    for x in range(inputData.shape[0]):
        for y in range(inputData.shape[1]):
            ret[x+y].append(inputData[x][y])
    return ret








if __name__ == '__main__':
    exit(0)

