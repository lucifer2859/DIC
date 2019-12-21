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

# 

class RecNet(nn.Module):
    def __init__(self):
        super(RecNet, self).__init__()

        self.conv_channels_up = nn.Conv2d(3, 64, 1)

        convList = []
        for i in range(32):
            convList.append(baseNet.ResNet(transpose=False, channels=64, kernel_size=3, padding=1, skip=False))
        self.convList = nn.Sequential(*convList)

        self.conv_channels_down = nn.Conv2d(64, 3, 1)


    def forward(self, x):
        return F.leaky_relu_(self.conv_channels_down(self.convList(F.leaky_relu_(self.conv_channels_up(x)))))


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
    recNet = RecNet().cuda().train()
    print('create new model')
else:
    recNet = torch.load('../models/recNet_' + sys.argv[0] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda().train()
    print('read ../models/' + sys.argv[0] + '.pkl')

# print(recNet)

optimizer = torch.optim.Adam([{'params':recNet.parameters()}], lr=float(sys.argv[3]))

trainData = torch.empty([batchSize, 3, 256, 256]).float().cuda()

lastSavedI = 0
C = torch.arange(start=0, end=16).unsqueeze_(1).float().cuda()
sigma = 1
testImgSum = 24
testImgDir = '/home/dchen/dataset/256bmp/test/'
testDataReader = bmpLoader.datasetReader(colorFlag=True, batchSize=1, bufferBatchSizeMultiple=testImgSum,
                                         imgDir=testImgDir, imgSum=testImgSum)
testData = torch.empty([testImgSum, 3, 256, 256]).float().cuda()
for k in range(testImgSum):
    testData[k] = torch.from_numpy(testDataReader.readImg()).float().cuda()


for i in range(100000):

    for k in range(batchSize):
        trainData[k] = torch.from_numpy(dReader.readImg()).float().cuda()

    if (i == 0):
        trainDataRgbFilter = rgbCompress.createRGGB121Filter(trainData).cuda()
        testDataRgbFilter = rgbCompress.createRGGB121Filter(testData).cuda()


    optimizer.zero_grad()
    decData = recNet(trainData * trainDataRgbFilter)

    currentMSEL = F.mse_loss(decData, trainData)

    currentMS_SSIM = pytorch_msssim.ms_ssim(trainData, decData, data_range=255, size_average=True)

    if (torch.isnan(currentMS_SSIM)):
        currentMS_SSIM.zero_()
    loss = - currentMS_SSIM.item()

    # print('%.3f' % currentMS_SSIM.item(), '%.3f' % currentMSEL.item())

    if (currentMS_SSIM.item() > -0.7):
        currentMSEL.backward()
    else:
        currentMS_SSIM.backward()

    if (i == 0):
        minLoss = loss
    else:
        if (minLoss > loss and loss < -0.9 ):  # 保存最小loss对应的模型
            # 在测试一组数据进行检测
            for param in recNet.parameters():
                param.requires_grad = False

            decData = recNet(testData * testDataRgbFilter)
            testLoss = - pytorch_msssim.ms_ssim(testData, decData, data_range=255,
                                                size_average=True).item()
            
            print('test:%d,%.3f' % (i, testLoss))

            for param in recNet.parameters():
                param.requires_grad = True

            if (minLoss > testLoss):  # 保存最小loss对应的模型
                minLoss = testLoss
                torch.save(recNet, '../models/recNet_' + sys.argv[0] + '.pkl')
                print('save ../models/' + sys.argv[0] + '.pkl')
                lastSavedI = i
            print('minTestLoss:%d,%.3f' % (lastSavedI, minLoss))


    optimizer.step()

    # print(sys.argv)
    # print('训练到第', i, '次')
    # print('本次训练loss=', '%.3f' % loss)
    # print('minLoss=', '%.3f' % minLoss, '上次保存模型时对应的训练次数为', lastSavedI)

    print('train:%d,%.3f' % (i, loss))
