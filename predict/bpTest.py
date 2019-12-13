
import torch
import sys
import os
sys.path.append('../')
import npyHuffmanEncoder
import blockPredictNet
from PIL import Image
import numpy
def tensorToImage(inputData):
    inputDataNpy = inputData.numpy().astype('uint8').reshape(inputData.shape[0], inputData.shape[1])
    return Image.fromarray(inputDataNpy)

cudaDevice = '1'
blockSize = 2

torch.cuda.set_device(int(cudaDevice)) # 设置使用哪个显卡
bpNet = torch.load('../models/bpNet_p5.py.pkl', map_location='cuda:' + cudaDevice).cuda()
for param in bpNet.parameters():
    param.requires_grad = False

img = Image.open('../lena.bmp').convert('L')
testData = torch.from_numpy(numpy.asarray(img).astype(float)
                            .reshape([1, 512, 512])).reshape([1, 1, 512, 512]).float().cuda()
pdData = torch.zeros_like(testData) # 预测数据
testDataBlock = torch.zeros(size=[1, 3, blockSize, blockSize], device=testData.device)

for yy in range(0, 512, blockSize):
    for xx in range(0, 512, blockSize):
        if(xx % 128==0 and yy % 128==0):
            print(xx,yy)
        if(xx==0 or yy==0):
            # 左上边界的块完整保存
            pdData[:, :, yy:yy+blockSize, xx:xx+blockSize] = testData[:, :, yy:yy+blockSize, xx:xx+blockSize].clone()
        else:
            # 其余块用预测值
            testDataBlock[:, 0:1, :, :] = testData[:, :, yy - blockSize:yy, xx - blockSize:xx].clone()
            testDataBlock[:, 1:2, :, :] = testData[:, :, yy - blockSize:yy, xx:xx + blockSize].clone()
            testDataBlock[:, 2:3, :, :] = testData[:, :, yy:yy + blockSize, xx - blockSize:xx].clone()
            pdData[:, :, yy:yy+blockSize, xx:xx+blockSize] = bpNet(testDataBlock / 255) * 255


tensorToImage(pdData[0][0].cpu()).save('./p5.bmp')
print('save bmp')
resData = testData - pdData
resData[:, :, :, 0:blockSize] = testData[:, :, :, 0:blockSize].clone()
resData[:, :, 0:blockSize, :] = testData[:, :, 0:blockSize, :].clone()
numpy.save('./res.npy', resData[0].detach().cpu().numpy().astype(int))
print('save npy')
tensorToImage(resData[0][0].abs().cpu()).save('./p5_res.bmp')
npyHuffmanEncoder.main('./res.npy', 'res.b')
fileSize = os.path.getsize('./res.b')
print(fileSize)


