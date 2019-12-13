from PIL import Image
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import torch.optim
import sys
sys.path.append('../')
import os
import pytorch_msssim
import bmpLoader
import rgbCompress
import softToHardQuantize
import baseNet
import pixelSf.pixelShuffleFunction as pixelShuffleFunction
# 量化等级改为32


class EncodeNet(nn.Module):
    def __init__(self, channels):
        super(EncodeNet, self).__init__()

        self.conv_channels_up = nn.Conv2d(1, channels, 1)
        convList = []
        for i in range(4):
            convList.append(pixelShuffleFunction.PixelUnshuffle(2))
            convList.append(nn.Conv2d(channels*4, channels, 1))
            convList.append(nn.LeakyReLU(inplace=True))
            convList.append(nn.BatchNorm2d(channels))
            convList.append(baseNet.ResNet(transpose=False, channels=channels, kernel_size=3, padding=1))

        convList.append(nn.Conv2d(channels, 16, 1))
        convList.append(nn.LeakyReLU(inplace=True))

        self.convList = nn.Sequential(*convList)
    def forward(self, x):
        return self.convList(F.leaky_relu_(self.conv_channels_up(x)))

class DecodeNet(nn.Module):
    def __init__(self, channels):
        super(DecodeNet, self).__init__()
        self.conv_channels_down = nn.Conv2d(channels, 1, 1)
        convList = []
        convList.append(nn.Conv2d(16, channels, 1))
        convList.append(nn.LeakyReLU(inplace=True))
        for i in range(4):
            convList.append(baseNet.ResNet(transpose=False, channels=channels, kernel_size=3, padding=1))
            convList.append(nn.BatchNorm2d(channels))
            convList.append(nn.Conv2d(channels, channels*4, 1))
            convList.append(nn.LeakyReLU(inplace=True))
            convList.append(nn.PixelShuffle(2))

        self.convList = nn.Sequential(*convList)
    def forward(self, x):
        return F.leaky_relu_(self.conv_channels_down(self.convList(x)))


'''
argv:
1: 使用哪个显卡
2: 为0则重新开始训练 否则读取之前的模型
3: 学习率 Adam默认是1e-3
4: batchSize
'''

if(len(sys.argv)!=5):
    print('1: 使用哪个显卡\n'
          '2: 为0则重新开始训练 否则读取之前的模型\n'
          '3: 学习率 Adam默认是1e-3\n'
          '4: batchSize')
    exit(0)

batchSize = int(sys.argv[4]) # 一次读取?张图片进行训练
dReader = bmpLoader.datasetReader(colorFlag=True, batchSize=batchSize)
torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡


if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    encNet = EncodeNet(128).cuda().train()
    decNet = DecodeNet(128).cuda().train()
    print('create new model')
else:
    encNet = torch.load('../models/encNet_' + sys.argv[0] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda().train()
    decNet = torch.load('../models/decNet_' + sys.argv[0] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda().train()
    print('read ../models/' + sys.argv[0] + '.pkl')

print(encNet)
print(decNet)

optimizer = torch.optim.Adam([{'params':encNet.parameters()},{'params':decNet.parameters()}], lr=float(sys.argv[3]))

trainData = torch.empty([batchSize, 3, 256, 256]).float().cuda()

lastSavedI = 0
C = torch.arange(start=0, end=32).unsqueeze_(1).float().cuda()
sigma = 1
testImgSum = 24
testImgDir = '/datasets/MLG/wfang/imgCompress/kodim256/'
testDataReader = bmpLoader.datasetReader(colorFlag=True, batchSize=1, bufferBatchSizeMultiple=testImgSum,
                                         imgDir=testImgDir, imgSum=testImgSum)
testData = torch.empty([testImgSum, 3, 256, 256]).float().cuda()
for k in range(testImgSum):
    testData[k] = torch.from_numpy(testDataReader.readImg()).float().cuda()


for i in range(999999999):

    for k in range(batchSize):
        trainData[k] = torch.from_numpy(dReader.readImg()).float().cuda()

    if (i == 0):
        rgbKernelList = rgbCompress.createRGGB121ConvKernel()
        for j in range(rgbKernelList.__len__()):
            rgbKernelList[j] = rgbKernelList[j].cuda()
        testGrayData = rgbCompress.RGB444DetachToRGGBTensor(testData, rgbKernelList, True)
        print('read all testData')

    optimizer.zero_grad()
    trainGrayData = rgbCompress.RGB444DetachToRGGBTensor(trainData, rgbKernelList, True)
    encData = encNet(trainGrayData / 255)

    qEncData = torch.zeros_like(encData)
    for ii in range(encData.shape[0]):
        qEncData[ii] = softToHardQuantize.sthQuantize(encData[ii], C, sigma)

    decData = decNet(qEncData) * 255
    currentMSEL = F.mse_loss(trainGrayData, decData)

    currentMS_SSIM = pytorch_msssim.ms_ssim(trainGrayData, decData, win_size=7, data_range=255, size_average=True)

    if (torch.isnan(currentMS_SSIM)):
        currentMS_SSIM.zero_()
    loss = - currentMS_SSIM.item()

    print('%.3f' % currentMS_SSIM.item())

    if (currentMS_SSIM.item() > -0.7):
        currentMSEL.backward()
    else:
        currentMS_SSIM.backward()

    if (i == 0):
        minLoss = loss
    else:
        if (minLoss > loss):  # 保存最小loss对应的模型
            # 在测试一组数据进行检测
            for param in encNet.parameters():
                param.requires_grad = False
            for param in decNet.parameters():
                param.requires_grad = False

            encData = encNet(testGrayData / 255)
            qEncData = torch.zeros_like(encData)
            for ii in range(encData.shape[0]):
                qEncData[ii] = softToHardQuantize.sthQuantize(encData[ii], C, sigma)

            decData = decNet(qEncData) * 255
            testLoss = - pytorch_msssim.ms_ssim(testGrayData, decData, win_size=7, data_range=255,
                                                size_average=True).item()
            print('test loss', testLoss)
            for param in encNet.parameters():
                param.requires_grad = True
            for param in decNet.parameters():
                param.requires_grad = True
            if (minLoss > testLoss):  # 保存最小loss对应的模型
                minLoss = testLoss
                torch.save(encNet, '../models/encNet_' + sys.argv[0] + '.pkl')
                torch.save(decNet, '../models/decNet_' + sys.argv[0] + '.pkl')
                print('save ../models/' + sys.argv[0] + '.pkl')
                lastSavedI = i


    optimizer.step()

    print(sys.argv)
    print('训练到第',i,'次')
    print('本次训练loss=', '%.3f' % loss)
    print('minLoss=', '%.3f' % minLoss, '上次保存模型时对应的训练次数为',lastSavedI)
