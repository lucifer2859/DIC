# leinao
类脑智能课程大作业
### 本次类脑智能课程大作业主要的图像压缩模型文件参见rgb121Compress_new, rgbCom_new;
#### 在文件夹内直接执行如下命令python xxx.py [argv]
##### argv:
1. 使用哪个显卡;
2. 0表示重新开始训练, 否则读取之前的模型;
3. 学习率, Adam默认是1e-3;
4. batchSize;
##### e.g. python 0.py 0 0 1e-3 16
##### e.g. python rec1.py 0 0 1e-3 2

### 其他相关文件
#### baseNet.py: 基本网络结构的定义;
#### bmpLoader.py: 加载数据集（训练集）;
#### rgbCompress.py: 将RGB图片通过RGGB等形式进行处理;
#### pytorch_gdn: GDN层的pytorch实现;
#### pytorch_msssim: MS-SSIM的pytorch实现;
#### rgb121Compress: 之前科研过程中的图像压缩模型;
#### rgbCom: 之前科研过程中的ResNet复原模型;
#### models: 模型文件的存储地址;
#### log: 模型训练日志的存储地址;
#### log/data: 处理后的模型训练日志的存储地址.

### 移植到新环境中需要根据训练集和测试集地址修改bmpLoader.py与rgb121Compress_new/xxx.py中的对应内容.


