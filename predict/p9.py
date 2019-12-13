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
import blockPredictNet






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
dReader = bmpLoader.datasetReader(colorFlag=False, batchSize=batchSize, bufferBatchSizeMultiple=4)
torch.cuda.set_device(int(sys.argv[1])) # 设置使用哪个显卡
blockSize = 2

if(sys.argv[2]=='0'): # 设置是重新开始 还是继续训练
    bpNet = blockPredictNet.blockPredictNet(128, 2).cuda().train()
    print('create new model')
else:
    bpNet = torch.load('../models/bpNet_' + sys.argv[0] + '.pkl', map_location='cuda:'+sys.argv[1]).cuda().train()
    print('read ../models/' + sys.argv[0] + '.pkl')

print(bpNet)

optimizer = torch.optim.Adam([{'params':bpNet.parameters()}], lr=float(sys.argv[3]))

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
trainDataBlock = torch.zeros(size=[batchSize, 3, blockSize, blockSize], device=testData.device)
testDataBlock = torch.zeros(size=[testImgSum, 3, blockSize, blockSize], device=testData.device)
while(True):

    for k in range(batchSize):
        trainData[k] = torch.from_numpy(dReader.readImg()).float().cuda()

    for y in range(blockSize, 256, blockSize):
        for x in range(blockSize, 256, blockSize):
            optimizer.zero_grad()
            trainDataBlock[:, 0:1, :, :] = trainData[:, :, y - blockSize:y, x - blockSize:x].clone()
            trainDataBlock[:, 1:2, :, :] = trainData[:, :, y - blockSize:y, x:x + blockSize].clone()
            trainDataBlock[:, 2:3, :, :] = trainData[:, :, y:y + blockSize, x - blockSize:x].clone()
            reData = trainData[:, :, y:y + blockSize, x:x + blockSize].clone()
            prData = bpNet(trainDataBlock / 255) * 255
            currentMSEL = F.mse_loss(prData, reData)

            loss = currentMSEL.item()
            currentMSEL.backward()

            if (i == 0):
                minLoss = loss
            else:
                if (minLoss > loss and loss < 1000):  # 保存最小loss对应的模型
                    # 在测试一组数据进行检测
                    print('test...')
                    for param in bpNet.parameters():
                        param.requires_grad = False

                    testLoss = 0
                    testTime = 0
                    for yy in range(blockSize, 256, blockSize * 2):
                        for xx in range(blockSize, 256, blockSize * 2):
                            testDataBlock[:, 0:1, :, :] = testData[:, :, yy - blockSize:yy,
                                                          xx - blockSize:xx].clone()
                            testDataBlock[:, 1:2, :, :] = testData[:, :, yy - blockSize:yy,
                                                          xx:xx + blockSize].clone()
                            testDataBlock[:, 2:3, :, :] = testData[:, :, yy:yy + blockSize,
                                                          xx - blockSize:xx].clone()
                            reData = testData[:, :, yy:yy + blockSize, xx:xx + blockSize].clone()
                            prData = bpNet(testDataBlock / 255) * 255

                            testLoss = testLoss + F.mse_loss(prData.round(), reData).item()
                            testTime = testTime + 1
                    testLoss = testLoss / testTime
                    print('test loss', testLoss)

                    for param in bpNet.parameters():
                        param.requires_grad = True
                    if (minLoss > testLoss):  # 保存最小loss对应的模型
                        minLoss = testLoss
                        torch.save(bpNet, '../models/bpNet_' + sys.argv[0] + '.pkl')
                        print('save ../models/' + sys.argv[0] + '.pkl')
                        lastSavedI = i

            optimizer.step()
            if (i % 2 == 0):
                print(sys.argv)
                print('训练到第', i, '次')
                print('本次训练loss=', '%.3f' % loss)
                print('minLoss=', '%.3f' % minLoss, '上次保存模型时对应的训练次数为', lastSavedI)
            i = i + 1


