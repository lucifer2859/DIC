from PIL import Image
import numpy
import torch
import torch.nn.functional as F

def create10100101Filter(inputData):
    '''
    inputData是偶数通道，创建滤波器以对其进行如下采样：
    0,2,4...通道
    [1 1    ->    [1 0
     1 1]          0 1]
    1,3,5...通道
    [1 1    ->    [0 1
     1 1]          1 0]
    :param inputData: batchSize * c * m * n
    :return: batchSize * c * m * n
    '''
    filter = torch.zeros(size=[2, 2, 2]).float()

    filter[0][0][0] = 1
    filter[0][1][1] = 1

    filter[1][0][1] = 1
    filter[1][1][0] = 1

    return filter.repeat(inputData.shape[1] // 2, inputData.shape[2] // 2, inputData.shape[3] // 2).expand(inputData.shape)

def create10101111Filter(inputData):
    '''
    inputData是偶数通道，创建滤波器以对其进行如下采样：
    0,2,4...通道
    [1 1    ->    [1 0
     1 1]          0 1]
    1,3,5...通道
    [1 1    ->    [1 1
     1 1]          1 1]
    :param inputData: batchSize * c * m * n
    :return: batchSize * c * m * n
    '''
    filter = torch.zeros(size=[2, 2, 2]).float()

    filter[0][0][0] = 1
    filter[0][1][1] = 1

    filter[1][0][0] = 1
    filter[1][0][1] = 1
    filter[1][1][0] = 1
    filter[1][1][1] = 1

    return filter.repeat(inputData.shape[1] // 2, inputData.shape[2] // 2, inputData.shape[3] // 2).expand(inputData.shape)

def createRGGB121Filter(inputData):
    '''
    创建一个用于将RGB444转换成RGB121的滤波器
    :param inputData: batchSize * 3 * m * n
    :return: batchSize * 3 * m * n
    '''
    RGBFilter = torch.zeros(size=[3, 2, 2]).float()
    RGBFilter[0][0][0] = 1  # R
    RGBFilter[1][0][1] = 1  # G
    RGBFilter[1][1][0] = 1  # G
    RGBFilter[2][1][1] = 1  # B
    return RGBFilter.repeat(1, inputData.shape[2]//2, inputData.shape[3]//2).expand(inputData.shape)

def createRRGB211Filter(inputData):
    '''
    创建一个用于将RGB444转换成RRGB的滤波器
    R
    [0 1 0 1
     1 0 1 0
     0 1 0 1
     1 0 1 0]
    G
    [1 0 0 0
     0 1 0 0
     0 0 1 0
     0 0 0 1]
    B
    [0 0 1 0
     0 0 0 1
     1 0 0 0
     0 1 0 0]
    :param inputData: batchSize * 3 * m * n
    :return: batchSize * 3 * m * n
    '''
    RGBFilter = torch.zeros(size=[3, 4, 4]).float()
    for i in range(4):
        for j in range(4):
            if((i+j)%2!=0):
                RGBFilter[0][i][j] = 1 # R

    RGBFilter[1][0][0] = 1  # G
    RGBFilter[1][1][1] = 1  # G
    RGBFilter[1][2][2] = 1  # G
    RGBFilter[1][3][3] = 1  # G

    RGBFilter[2][0][2] = 1  # B
    RGBFilter[2][1][3] = 1  # B
    RGBFilter[2][2][0] = 1  # B
    RGBFilter[2][3][1] = 1  # B
    return RGBFilter.repeat(1, inputData.shape[2] // 4, inputData.shape[3] // 4).expand(inputData.shape)

def RGB444ToRGGB121(inputData, RGBFilter):
    '''
    RGB444图像转换成RGGB121
    [RGB RGB    ->   [R G
     RGB RGB]         G B]
    :param inputData: batchSize * 3 * m * n
    :param RGBFilter: batchSize * 3 * m * n
    :return: batchSize * 1 * m * n
    '''
    RGGB121Data = (inputData * RGBFilter).permute([1,2,3,0]) # 3 * m * n * batchSize
    RGGB121Data = RGGB121Data[0] + RGGB121Data[1] + RGGB121Data[2] # m * n * batchSize
    RGGB121Data.unsqueeze_(0) # 1 * m * n * batchSize
    return RGGB121Data.permute([3,0,1,2])

def RGB121ToRGGB444Shape(inputData):
    '''
    RGB121采样得到的灰度图，尺寸转换为原来彩色图的尺寸，但0位置不会被填充
    :param inputData: batchSize * 1 * m * n
    :return: batchSize * 3 * m * n
    '''
    ret = torch.zeros(size=[inputData.shape[0], 3, inputData.shape[2], inputData.shape[3]])
    for i in range(inputData.shape[0]):
        for x in range(inputData.shape[2]):
            for y in range(inputData.shape[3]):
                if(x%2==0 and y%2==0):
                    ret[i][0][x][y] = inputData[i][0][x][y]
                elif(x%2==1 and y%2==1):
                    ret[i][2][x][y] = inputData[i][0][x][y]
                else:
                    ret[i][1][x][y] = inputData[i][0][x][y]

    return ret

def createRGGB121ConvKernel():
    RKernel = torch.zeros(size=[1, 1, 2, 2]).float()
    G1Kernel = torch.zeros(size=[1, 1, 2, 2]).float()
    G2Kernel = torch.zeros(size=[1, 1, 2, 2]).float()
    BKernel = torch.zeros(size=[1, 1, 2, 2]).float()
    RKernel[0][0][0][0] = 1
    G1Kernel[0][0][0][1] = 1
    G2Kernel[0][0][1][0] = 1
    BKernel[0][0][1][1] = 1
    return [RKernel, G1Kernel, G2Kernel, BKernel]

def RGB444DetachToRGGBTensor(inputData, KernelList, catFlat = False, catDim=0, retType = None):
    '''
    [RGB RGB    ->   [ [R] [G] [G] [B] ]
     RGB RGB]
    inputData: batchSize * 3 * m * n
    '''
    inputData = inputData.permute([1,2,3,0]) # 3 * m * n * batchSize

    if(catFlat == False and retType!=None):
        if(retType=='R'):
            return F.conv2d(inputData[0].permute([2, 0, 1]).unsqueeze(1), KernelList[0], stride=2)
        elif(retType=='G1'):
            return F.conv2d(inputData[1].permute([2, 0, 1]).unsqueeze(1), KernelList[1], stride=2)
        elif(retType=='G2'):
            return F.conv2d(inputData[1].permute([2, 0, 1]).unsqueeze(1), KernelList[2], stride=2)
        elif(retType=='B'):
            return F.conv2d(inputData[2].permute([2, 0, 1]).unsqueeze(1), KernelList[3], stride=2)

    # (m,n,batchSize) -> (batchSize,m,n) -> (batchSize,1,m,n) -> (batchSize,1,m/2,n/2)
    RTensor = F.conv2d(inputData[0].permute([2, 0, 1]).unsqueeze(1), KernelList[0], stride=2) 
    G1Tensor = F.conv2d(inputData[1].permute([2, 0, 1]).unsqueeze(1), KernelList[1], stride=2)
    G2Tensor = F.conv2d(inputData[1].permute([2, 0, 1]).unsqueeze(1), KernelList[2], stride=2)
    BTensor = F.conv2d(inputData[2].permute([2, 0, 1]).unsqueeze(1), KernelList[3], stride=2)
    if(catFlat==True):
        # (batchSize,1,m/2,n/2) -> (4*batchSize,1,m/2,n/2)
        return torch.cat((RTensor, G1Tensor, G2Tensor, BTensor), catDim)
    elif(catFlat==False):
        return RTensor, G1Tensor, G2Tensor, BTensor


def RGGBTensorToRGB444(inputData):
    '''
    :param inputData: 4 * 1 * m * n
    :return: 1 * 3 * 2m * 2n
    def RGB444DetachToRGGBTensor(inputData, KernelList, catFlat = False)的逆操作
    '''
    ret = torch.zeros(size=[1, 3, 2*inputData.shape[2], 2*inputData.shape[2]])

    for i in range(ret.shape[2]):
        for j in range(ret.shape[3]):
            if(i%2==0 and j%2==0):#R
                ret[0][0][i][j] = inputData[0][0][i // 2][j // 2]
            elif(i%2==1 and j%2==1):#G
                ret[0][2][i][j] = inputData[3][0][(i - 1) // 2][(j - 1) // 2]
            elif (i % 2 == 0 and j % 2 == 1): # B1
                ret[0][1][i][j] = inputData[1][0][i // 2][(j - 1) // 2]
            elif (i % 2 == 1 and j % 2 == 0):  # B2
                ret[0][1][i][j] = inputData[2][0][(i - 1) // 2][j // 2]

    return ret





