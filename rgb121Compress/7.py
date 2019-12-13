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
import autoQuantize

class HollowConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        assert kernel_size%2!=0, 'kernel_size%2!=0'
        super(HollowConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)
        self.mask = torch.ones(size=[out_channels, in_channels, kernel_size, kernel_size]).cuda()
        self.mask[:, :, kernel_size//2, kernel_size//2] = 0





    def forward(self, x):
        self.weight.data *= self.mask
        return super(HollowConv2d, self).forward(x)





class PredictCenterNet(nn.Module):
    def __init__(self):
        super(PredictCenterNet, self).__init__()
        '''
        batchSize * 1 * 128 * 128 -> batchSize * 1 * 128 * 128 * 256
        '''

        self.hc = HollowConv2d(1, 512, 3, padding=1)
        self.conv = nn.Conv2d(512, 256, 1)



    def forward(self, x):
        y = F.leaky_relu_(self.hc(x))
        # batchSize * 1 * 128 * 128 -> batchSize * 512 * 128 * 128
        y = F.leaky_relu_(self.conv(y))
        # batchSize * 512 * 128 * 128 -> batchSize * 256 * 128 * 128
        y = y.view(y.shape[0], 1, 256, y.shape[2], y.shape[3])
        # batchSize * 256 * 128 * 128 -> batchSize * 1 * 256 * 128 * 128
        y = y.permute(0, 1, 3, 4, 2)
        # batchSize * 1 * 256 * 128 * 128 -> batchSize * 1 * 128 * 128 * 256
        return y





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
    pcNet = PredictCenterNet().cuda().train()
    print('create new model')
else:
    pcNet = torch.load('../models/pcNet_' + sys.argv[0] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda().train()
    print('read ../models/' + sys.argv[0] + '.pkl')

print(pcNet)

optimizer = torch.optim.Adam([{'params':pcNet.parameters()}], lr=float(sys.argv[3]))
trainData = torch.empty([batchSize, 3, 256, 256]).float().cuda()

lastSavedI = 0
C = torch.arange(start=0, end=16).unsqueeze_(1).float().cuda()
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
    encData = pcNet(trainGrayData)

    loss = F.cross_entropy(encData.contiguous().view(-1, 256), trainGrayData.contiguous().view(-1).long())
    loss.backward()


    if (i == 0):
        minLoss = loss.item()
    else:
        if (minLoss > loss.item()):  # 保存最小loss对应的模型
            # 在测试一组数据进行检测
            for param in pcNet.parameters():
                param.requires_grad = False


            encData = pcNet(testGrayData)

            testLoss = F.cross_entropy(encData.contiguous().view(-1, 256), testGrayData.contiguous().view(-1).long()).item()
            print('test loss', testLoss)
            for param in pcNet.parameters():
                param.requires_grad = True

            if (minLoss > testLoss):  # 保存最小loss对应的模型
                minLoss = testLoss
                torch.save(pcNet, '../models/pcNet_' + sys.argv[0] + '.pkl')
                print('save ../models/' + sys.argv[0] + '.pkl')
                lastSavedI = i


    optimizer.step()

    print(sys.argv)
    print('训练到第',i,'次')
    print('本次训练loss=', '%.3f' % loss)
    print('minLoss=', '%.3f' % minLoss, '上次保存模型时对应的训练次数为',lastSavedI)
