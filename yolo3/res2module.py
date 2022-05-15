#Res2Net
from __future__ import division
import torch
import torch.nn as nn
import os
from torch.autograd import Variable

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
'''
用在YOLOv3-Tiny中的Res2Net模块：只增加输出特征图的通道，不改变其尺寸。
'''
class Res2Block(nn.Module):
    def __init__(self, features_size, stride_ = 1, scale = 4, padding_ = 1, groups_ = 1, expand = 2):
        super(Res2Block,self).__init__()
        #erro for wrong input如果输入不正确则会报错
        # features_size = 64
        # self.expand = 2
        if scale < 2 or features_size % scale:
            print('Error:illegal input for scale or feature size')

        # self.divided_features = 16
        self.divided_features = int(expand*features_size / scale) # 将输入特征分成N个部分

        self.relu = nn.ReLU(inplace=True)
        # res_block支路上的1*1卷积，确定该Res2Net是通道增加还是减少；
        # 用来调整原始输入特征图的通道数
        self.conv1 = nn.Conv2d(features_size, expand*features_size, kernel_size=1, stride=stride_, padding=0, groups=groups_)
        self.bn1 = nn.BatchNorm2d(expand*features_size)
        # 支路连接，扩充样本通道
        self.conv1_1 = nn.Conv2d(expand*features_size, expand*features_size, kernel_size=1, stride=stride_, padding=0, groups=groups_)
        self.bn1_1 = nn.BatchNorm2d(expand*features_size)
        # 内部分为多个块以后的卷积操作
        self.conv2 = nn.Conv2d(self.divided_features, self.divided_features, kernel_size=3, stride=stride_, padding=padding_, groups=groups_)
        self.bn2 = nn.BatchNorm2d(self.divided_features)
        # 将convs定义为包括多个卷积模型的存储器（存储模块）
        self.convs = nn.ModuleList()
        # scale - 2 = 2循环执行两次 除了原始保留的（不做任何卷积操作）和第二个分支卷积（不承接前一个卷积输出的），剩余的所有分支
        for i in range(scale - 2):

            self.convs.append(
                nn.Conv2d(self.divided_features, self.divided_features, kernel_size=3, stride=stride_, padding=padding_, groups=groups_)
            )

    # 构造Res2Net模块
    def forward(self, x):
        # x为输入特征
        # features_in.shape = torch.Size([8, 64, 32, 32])
        features_in = x
        # 这次卷积为res2模块前的那一次卷积，可以用来调整通道数，是否需要1x1卷积层根据自己网络的情况而定
        # conv1_out.shape = torch.Size([8, 64, 32, 32])
        conv1_out = self.conv1(features_in) # 第一个1*1卷的输出
        conv1_out = self.bn1(conv1_out)
        conv1_out = self.relu(conv1_out)
        # y1为res2模块中的第一次卷积（特征没变，所以相当于没做卷积）
        # y1.shape = torch.Size([8, 16, 32, 32])
        y1 = conv1_out[:,0:self.divided_features,:,:]
        # y2.shape = torch.Size([8, 16, 32, 32])
        y2 = conv1_out[:,self.divided_features:2*self.divided_features,:,:]
        # fea为res2模块中的第二次卷积，下面用features承接了
        fea = self.conv2(y2) # 真实的y2
        fea = self.bn2(fea)
        fea = self.relu(fea)
        # 第二次卷积后的特征
        # 这里之所以用features变量承接是因为方便将后三次的卷积结果与第一次的卷积结果做拼接
        # features.shape = torch.Size([8, 16, 32, 32])
        features = fea
        # 在进行后续连接时，先将特征层初始化，
        # features、fea都定义为y2
        # self.convs中只有两层网络
        for i, conv in enumerate(self.convs):
            # 第一次循环pos = 16 i从1开始遍历？？
            # 第二次循环pos = 32
            pos = (i + 1)*self.divided_features
            # 第一次循环divided_feature.shape = torch.Size([8, 16, 32, 32])
            # 第二次循环divided_feature.shape = torch.Size([8, 16, 32, 32])
            divided_feature = conv1_out[:,pos:pos+self.divided_features,:,:]
            # 第三次和第四次卷积就是这行代码
            # 将上一次卷积结果与本次卷积的输入拼接后作为新的输入特征
            fea = conv(fea + divided_feature)
            fea = self.bn2(fea)
            fea = self.relu(fea)
            # 下面这行代码是在此for循环完成后将后三次卷积的结果拼接在一起
            features = torch.cat([features, fea], dim = 1)
        # 将第一次的卷积和后三次卷积的结果做拼接
        out = torch.cat([y1, features], dim = 1)
        # 对拼接后的特征做1x1卷积，调整通道数
        conv1_out1 = self.relu(self.bn1_1(self.conv1_1(out)))
        # 在YOLOv3-Tiny中只需让Res2Net module实现增加通道数的目的即可
        feature_res = self.relu(self.bn1(self.conv1(features_in)))
        features_out = conv1_out1 + feature_res
        # 输出特征
        return features_out

if __name__ == "__main__":
    res2block = Res2Block(64,1,4,1,1,2)
    res2block.cuda()
    # bs,channels,height,width
    x = Variable(torch.rand([8, 64, 32, 32]).cuda())
    y = res2block(x)
    # x.shape = torch.Size([8, 64, 32, 32])
    print(x.shape)
    # y.shape = torch.Size([8, 64, 32, 32])
    print(y.shape)
    print(res2block)
    torch.save(res2block, 'Res2Net.pth')
