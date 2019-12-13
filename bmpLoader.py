from PIL import Image
import numpy
import time
class datasetReader:

    imgDir = '/home/dchen/dataset/256bmp/train/'
    imgSumInMemory = 0 # 提前从硬盘读取，保存在内存中的图片数量
    trainedImgNum = 0 # 已经被读取进入gpu的图片数量
    imgData = numpy.array([0])

    imgSum = 4000 # 一共有4000张图片
    readSeq = numpy.array([0]) # 要读取的图片标号
    colorFlag = False

    def __init__(self, colorFlag, batchSize, bufferBatchSizeMultiple = 256, imgDir = '/home/dchen/dataset/256bmp/train/', imgSum = 4000):
        '''
        :param colorFlag: 是否使用彩色图片，False时会将图片转化成灰度图
        :param batchSize: 一次训练中从内存中读入GPU进行训练的图片数量
        :param bufferBatchSizeMultiple: 从硬盘中读取bufferBatchSizeMultiple*batchSize数量的图片到内存当作中转数据
        :param imgDir:图片所在目录，图片需要按照0.bmp，1.bmp之类的格式命名
        :param imgSum:图片的总数
        '''
        numpy.random.seed()
        self.imgDir = imgDir
        self.imgSum = imgSum
        self.imgSumInMemory = min(bufferBatchSizeMultiple * batchSize, imgSum)
        self.colorFlag = colorFlag
        testImg = Image.open(self.imgDir + str(0) + '.bmp')
        self.imgW = testImg.size[0]
        self.imgH = testImg.size[1]

        if(self.colorFlag==True):
            self.imgData = numpy.empty([self.imgSumInMemory, 3, self.imgW, self.imgH])
        elif(self.colorFlag==False):
            self.imgData = numpy.empty([self.imgSumInMemory, 1, self.imgW, self.imgH])
        self.readImgToMemory()


    def readImgToMemory(self):
        startTime = time.time()
        print('\n开始读取', self.imgSumInMemory, '张图片到内存')
        self.readSeq = numpy.random.randint(low=0, high=self.imgSum, size=[self.imgSumInMemory])
        for i in range(self.imgSumInMemory):
            if (self.colorFlag == True):
                img = Image.open(self.imgDir + str(self.readSeq[i]) + '.bmp')
                self.imgData[i] = numpy.asarray(img).astype(float).transpose((2, 1, 0))
            elif(self.colorFlag == False):
                img = Image.open(self.imgDir + str(self.readSeq[i]) + '.bmp').convert('L')
                self.imgData[i] = numpy.asarray(img).astype(float).reshape([1, self.imgW, self.imgH])
        usedTime = time.time() - startTime
        print('用时',usedTime,'平均每张图片用时',usedTime/self.imgSumInMemory)
        if (self.colorFlag == True):
            print('大约占用', self.imgSumInMemory * self.imgW * self.imgH * 3 / 1024 / 1024, 'MB内存')
        elif (self.colorFlag == False):
            print('大约占用', self.imgSumInMemory * self.imgW * self.imgH / 1024 / 1024, 'MB内存')

    def readImg(self):
        retData = self.imgData[self.trainedImgNum]
        self.trainedImgNum = self.trainedImgNum + 1
        if(self.trainedImgNum == self.imgSumInMemory):
            self.readImgToMemory()
            self.trainedImgNum = 0
        return retData


def splitImage(img, size):
    '''
    将一张大图片分割成尺寸为size的多张小图片，返回一个list
    :param img: 输入图片
    :param size: 要分割成的尺寸，例如[16,16]
    :return: 元素是小图片的list
    '''
    ret = []
    for y in range(0, img.size[0], size[0]):
        for x in range(0, img.size[1], size[1]):
            ret.append(img.crop((x, y, x + size[0], y + size[1])))
    return ret

if __name__ == '__main__':
    img = Image.open('./test.bmp')
    smallList = splitImage(img, [16, 16])
    for i in range(smallList.__len__()):
        smallList[i].save('./img/' + str(i) + '.bmp')










