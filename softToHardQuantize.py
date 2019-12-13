import torch
import torch.nn.functional as nnF
class NearestQuantize(torch.autograd.Function): # 最近邻量化函数
    @staticmethod
    def forward(ctx, input, C):
        '''
        C: shape=[L, 1]的tensor
        input可以是批量的数据
        '''
        x = input.view(-1).repeat(C.shape[0], 1)
        dTensor = torch.abs(x - C)
        # dTensor[i]是与C[i]的距离
        return C[dTensor.argmin(0)].t().reshape(input.shape)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None # 把量化器的导数当做1

class SoftToHardQuantize(torch.nn.Module): # 量化函数
    def __init__(self):
        super(SoftToHardQuantize, self).__init__()

    '''
    量化取值集合
    C = {c_0, c_2, ..., c_(L-1)}，共L个取值
    对于一个标量值x，量化后为y

    前向传播时
    y = c_i
    i = arg min d(x, c_i)
    其中d(x, c_i) = ||c_i - x||

    记
    f(x, i) = exp(-sigma*d(x, c_i))
    则按照前向传播的计算公式
    sumD = f(x, 0) + f(x, 1) + ... + f(x, L-1)
    y = ( f(x, 0)*c_0 + f(x, 1)*c_1 + ... + f(x, L-1)*c_(L-1) ) / sumD
    用上述计算式子的导数来计算反向传播


    '''
    def forward(ctx, input, C, sigma):
        '''
        C: shape=[L, 1]的tensor
        input应该是单张图片，不能是批量的

        example:
        sigma = 1

        input
tensor([0, 5, 0, 2, 1, 7, 6, 1])

        C
tensor([[0.],
        [1.],
        [2.],
        [3.]])

        x = input.view(-1).repeat(C.shape[0], 1)

        x
tensor([[0., 5., 0., 2., 1., 7., 6., 1.],
        [0., 5., 0., 2., 1., 7., 6., 1.],
        [0., 5., 0., 2., 1., 7., 6., 1.],
        [0., 5., 0., 2., 1., 7., 6., 1.]])

        dTensor = -sigma * torch.pow((x - C), 2)

        dTensor
tensor([[ -0., -25.,  -0.,  -4.,  -1., -49., -36.,  -1.],
        [ -1., -16.,  -1.,  -1.,  -0., -36., -25.,  -0.],
        [ -4.,  -9.,  -4.,  -0.,  -1., -25., -16.,  -1.],
        [ -9.,  -4.,  -9.,  -1.,  -4., -16.,  -9.,  -4.]])

        y = torch.nn.functional.softmax(dTensor, 0)

        y
tensor([[7.2133e-01, 7.5318e-10, 7.2133e-01, 1.0442e-02, 2.0973e-01, 4.6583e-15, 1.8778e-12, 2.0973e-01],
        [2.6536e-01, 6.1031e-06, 2.6536e-01, 2.0973e-01, 5.7010e-01, 2.0609e-09, 1.1243e-07, 5.7010e-01],
        [1.3212e-02, 6.6928e-03, 1.3212e-02, 5.7010e-01, 2.0973e-01, 1.2339e-04, 9.1105e-04, 2.0973e-01],
        [8.9020e-05, 9.9330e-01, 8.9020e-05, 2.0973e-01, 1.0442e-02, 9.9988e-01, 9.9909e-01, 1.0442e-02]]) # 每一列的和均为1

        y = y * C

        y
tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
        [2.6536e-01, 6.1031e-06, 2.6536e-01, 2.0973e-01, 5.7010e-01, 2.0609e-09, 1.1243e-07, 5.7010e-01],
        [2.6423e-02, 1.3386e-02, 2.6423e-02, 1.1402e+00, 4.1946e-01, 2.4679e-04, 1.8221e-03, 4.1946e-01],
        [2.6706e-04, 2.9799e+00, 2.6706e-04, 6.2919e-01, 3.1325e-02, 2.9996e+00, 2.9973e+00, 3.1325e-02]])

        y = y.sum(0)

        y
tensor([0.2921, 2.9933, 0.2921, 1.9791, 1.0209, 2.9999, 2.9991, 1.0209])

        '''
        x = input.view(-1).repeat(C.shape[0], 1)
        # dTensor[i]是与C[i]的距离
        dTensor = -sigma * torch.pow((x - C), 2)

        y = torch.sum(nnF.softmax(dTensor, 0) * C, 0)

        return ((NearestQuantize.apply(input.view(-1), C) - y).detach_() + y).reshape(input.shape)
        #return y.reshape(input.shape)

sthQuantize = SoftToHardQuantize()

