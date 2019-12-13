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
import softToHardQuantize
import baseNet
# 4层




class EncodeNet(nn.Module):
    def __init__(self, channelsList):
        super(EncodeNet, self).__init__()
        self.conv_channels_up = nn.Conv2d(1, channelsList[0], 1)
        convList = []
        for i in range(channelsList.__len__() - 1):
            convList.append(baseNet.SampleNet(downSample=True, in_channels=channelsList[i], out_channels=channelsList[i+1]))
            print(channelsList[i], channelsList[i+1])

        self.convList = nn.Sequential(*convList)
    def forward(self, x):
        return self.convList(F.leaky_relu_(self.conv_channels_up(x)))

class DecodeNet(nn.Module):
    def __init__(self, channelsList):
        super(DecodeNet, self).__init__()
        self.tconv_channels_down = nn.ConvTranspose2d(channelsList[-1], 1, 1)
        convList = []
        for i in range(channelsList.__len__() - 1):
            convList.append(
                baseNet.SampleNet(downSample=False, in_channels=channelsList[i], out_channels=channelsList[i + 1]))
            print(channelsList[i], channelsList[i + 1])

        self.convList = nn.Sequential(*convList)
    def forward(self, x):
        return F.leaky_relu_(self.tconv_channels_down(self.convList(x)))


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
dReader = bmpLoader.datasetReader(colorFlag=False, batchSize=batchSize)
torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡
blockSize = 32

if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    channelsList = [1024, 512, 256, 128, 16]
    encNet = EncodeNet(channelsList).cuda().train()
    channelsList.reverse()
    decNet = DecodeNet(channelsList).cuda().train()
    print('create new model')
else:
    encNet = torch.load('../models/encNet_' + sys.argv[0] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda().train()
    decNet = torch.load('../models/decNet_' + sys.argv[0] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda().train()
    print('read ../models/' + sys.argv[0] + '.pkl')

print(encNet)
print(decNet)

optimizer = torch.optim.Adam([{'params':encNet.parameters()},{'params':decNet.parameters()}], lr=float(sys.argv[3]))

trainData = torch.empty([batchSize, 1, 256, 256]).float().cuda()

lastSavedI = 0
C = torch.arange(start=0, end=16).unsqueeze_(1).float().cuda()
sigma = 1
testImgSum = 24
testImgDir = '/datasets/MLG/wfang/imgCompress/kodim256/'
testDataReader = bmpLoader.datasetReader(colorFlag=False, batchSize=1, bufferBatchSizeMultiple=testImgSum,
                                         imgDir=testImgDir, imgSum=testImgSum)
testData = torch.empty([testImgSum, 1, 256, 256]).float().cuda()
for k in range(testImgSum):
    testData[k] = torch.from_numpy(testDataReader.readImg()).float().cuda()

i = 0
while(True):

    for k in range(batchSize):
        trainData[k] = torch.from_numpy(dReader.readImg()).float().cuda()

    for y in range(0, 256, blockSize):
        for x in range(0, 256, blockSize):
            optimizer.zero_grad()
            trainDataBlock = trainData[:, :, y:y+blockSize, x:x+blockSize]
            encData = encNet(trainDataBlock / 255)

            qEncData = torch.zeros_like(encData)
            for ii in range(encData.shape[0]):
                qEncData[ii] = softToHardQuantize.sthQuantize(encData[ii], C, sigma)

            decData = decNet(qEncData) * 255
            currentMSEL = F.mse_loss(trainDataBlock, decData)

            loss = currentMSEL.item()

            currentMSEL.backward()


            if (i == 0):
                minLoss = loss
            else:
                if (minLoss > loss and loss < 1000):  # 保存最小loss对应的模型
                    # 在测试一组数据进行检测
                    for param in encNet.parameters():
                        param.requires_grad = False
                    for param in decNet.parameters():
                        param.requires_grad = False

                    testLoss = 0
                    testTime = 0
                    for yy in range(0, 256, blockSize):
                        for xx in range(0, 256, blockSize):
                            testDataBlock = testData[:, :, yy:yy+blockSize, xx:xx+blockSize]
                            encData = encNet(testDataBlock / 255)
                            qEncData = torch.zeros_like(encData)
                            for ii in range(encData.shape[0]):
                                qEncData[ii] = softToHardQuantize.sthQuantize(encData[ii], C, sigma)

                            decData = decNet(qEncData) * 255
                            testLoss = testLoss + F.mse_loss(testDataBlock, decData).item()
                            testTime = testTime + 1
                    testLoss = testLoss / testTime
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
            if(i%32==0):
                print(sys.argv)
                print('训练到第', i, '次')
                print('本次训练loss=', '%.3f' % loss)
                print('minLoss=', '%.3f' % minLoss, '上次保存模型时对应的训练次数为', lastSavedI)
            i = i + 1


