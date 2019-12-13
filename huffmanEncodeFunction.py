import numpy
import torch
from bitstream import BitStream
from collections import namedtuple
from collections import OrderedDict
from queue import Queue, PriorityQueue

def unsignedValueToBinList(unsignedValue):
    strList = bin(unsignedValue)
    bList = []
    for i in range(2, strList.__len__()):
        if(strList[i]=='0'):
            bList.append(0)
        else:
            bList.append(1)

    return bList




class binTreeNode:
    def __init__(self, v, f):
        self.left = None
        self.right = None
        self.value = v
        self.frequency = f

    def __lt__(self, other):
        return self.frequency < other.frequency


def traversalTreeCreateTable(treeNode, huffmanTable, binList):
    if(treeNode.left!=None):
        binList.append(0)
        traversalTreeCreateTable(treeNode.left, huffmanTable, binList)
        binList.pop()

    if(treeNode.right!=None):
        binList.append(1)
        traversalTreeCreateTable(treeNode.right, huffmanTable, binList)
        binList.pop()

    if(treeNode.value!=None): # 防止重复访问
        huffmanTable.setdefault(treeNode.value, []).extend(binList)
        treeNode.value = None # 防止重复访问









def valueHuffmanEncode(inputData, bitStream):# inputData是一维numpy
    # 获取数据分布
    minV = int(inputData.min())
    maxV = int(inputData.max())
    inputDataHistc = torch.histc(torch.from_numpy(inputData).cuda(), min=minV, max=maxV,
                               bins=(maxV - minV + 1)).cpu().numpy()
    #print('inputData的最值，分布分别为', minV, maxV, inputDataHistc)

    vfQueue = PriorityQueue()# 数值-频数 优先队列
    vfQueue2 = PriorityQueue()# 用来保存频率排列，写入二进制文件用

    for i in range(minV, maxV + 1):
        if(inputDataHistc[i - minV]!=0):
            vfQueue.put(binTreeNode(i, inputDataHistc[i - minV]))
            vfQueue2.put(binTreeNode(i, inputDataHistc[i - minV]))
    # vfQueue中，按照频率从低到高排序，vfQueue.get()是频率最低的

    while True:
        # 选取2个最小的
        leftNode = vfQueue.get()
        rightNode = vfQueue.get()
        newNode = binTreeNode(None, leftNode.frequency + rightNode.frequency)
        newNode.left = leftNode
        newNode.right = rightNode
        vfQueue.put(newNode)
        if(vfQueue.qsize()==1):
            break

    rootNode = vfQueue.get()
    huffmanTable = {} # 字典，key为值，value为二进制码
    binList = []
    traversalTreeCreateTable(rootNode, huffmanTable, binList)
    #print('霍夫曼编码表为（类型为python 字典）', huffmanTable)
    # 保存霍夫曼表到文件---------------------------------------------------------
    bitStream.write(unsignedValueToBinList(minV), bool)  # 记录minV是多少
    bitStream.write(unsignedValueToBinList(maxV), bool) # 记录maxV是多少
    bitStream.write([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], bool)  # FF 分隔符

    # todo 这里其实改成只记录频率排列即可，因为解码器可以自己重建霍夫曼数
    '''
        for i in range(minV, maxV+1):
        #顺次记录 [minV, maxV]的二进制码
        if(inputDataHistc[i - minV]==0):
            bitStream.write([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], bool)  # 00 分隔符
        else:
            bitStream.write(huffmanTable[i], bool)
            bitStream.write([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], bool)  # FF 分隔符
    '''
    while True:
        if(vfQueue2.empty()==True):
            bitStream.write([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], bool)  # FF 分隔符
            break
        else:
            bitStream.write(vfQueue2.get().value, bool) # 只记录频率排列即可，因为解码器可以自己重建霍夫曼数
            bitStream.write([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], bool)  # FF 分隔符






    for i in range(inputData.shape[0]):
        bitStream.write(huffmanTable[int(inputData[i])], bool)

if __name__ == '__main__':
    exit(0)

