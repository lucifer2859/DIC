import dctTf
import rgbCompress
import torch
from PIL import Image
import numpy

def tensorToImage(inputData):
    return Image.fromarray(inputData.numpy().astype(int).reshape(inputData.shape[0], inputData.shape[1])).convert('L')


if __name__ == '__main__':

    torch.set_printoptions(sci_mode=False)
    testData = torch.from_numpy(numpy.asarray(Image.open('./test.bmp').convert('L')).astype(float).reshape([256, 256])).float().cuda()
    DCTKernel = torch.from_numpy(dctTf.createDCTKernel(256)).float().cuda()
    imgSum = 0
    LRFilter = dctTf.createLRFrequencyFilter(256, testData.device, 120, 240).cuda()
    ret = dctTf.getImgFrequencyComponent(testData, LRFilter, DCTKernel)
    tensorToImage(ret.cpu()).save('./img/1.bmp')












