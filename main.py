import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from paddle.nn import functional as F
import random
from paddle.io import Dataset
from visualdl import LogWriter
from paddle.vision.transforms import transforms as T
import warnings
warnings.filterwarnings("ignore")
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np



class conv_3x3(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv_3x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(ch_out),
            nn.ReLU(),
            nn.Conv2D(ch_out, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# TDB下采样 输入(h,w,c) 输出(h/2,w/2,c)
class TDB_Block(nn.Layer):
    def __init__(self, in_channel, out_channel, output_size):  # outsizes
        super(TDB_Block, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2D(output_size)
        self.maxpool = nn.MaxPool2D(2, 2)
        self.conv = nn.Conv2D(in_channel, out_channel * 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2D(out_channel * 2)
        self.residual = nn.Conv2D(in_channel * 4, out_channel, kernel_size=1, stride=1, padding=0)
        self.norm2 = nn.BatchNorm2D(out_channel)
        self.act = nn.Silu()

    def forward(self, x):
        x1 = self.avgpool(x)
        x2 = self.maxpool(x)
        x3 = self.conv(x)
        x4 = paddle.concat(x=[x1, x2, x3], axis=1)
        x = self.residual(x4)
        x = self.norm2(x)
        x = self.act(x)

        return x


# TUB上采样 输入(h,w,c) 输出(2h,2w,c)
class TUB_Block(nn.Layer):
    def __init__(self, in_channel, out_channel):
        super(TUB_Block, self).__init__()
        self.conv = nn.Conv2DTranspose(in_channel, out_channel, kernel_size=2, stride=2, padding=0)
        self.residual = nn.Conv2D(in_channel + out_channel, out_channel * 2, kernel_size=1)
        self.up_1 = nn.UpsamplingBilinear2D(scale_factor=2)
        self.norm1 = nn.BatchNorm2D(out_channel)
        self.norm2 = nn.BatchNorm2D(in_channel)
        self.act = nn.Silu()

    def forward(self, x):
        y = self.conv(x)
        x1 = self.norm1(y)
        z = self.up_1(x)
        x2 = self.norm2(z)
        x3 = paddle.concat(x=[x1, x2], axis=1)
        x4 = self.residual(x3)
        x = self.act(x4)
        return x


# SA
class SA(nn.Layer):

    def __init__(self, channel, groups=16):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.cweight = self.create_parameter((1, channel // (2 * groups), 1, 1),
                                             default_initializer=nn.initializer.Assign(
                                                 paddle.zeros((1, channel // (2 * groups), 1, 1))))
        self.cbias = self.create_parameter((1, channel // (2 * groups), 1, 1),
                                           default_initializer=nn.initializer.Assign(
                                               paddle.ones((1, channel // (2 * groups), 1, 1))))
        self.sweight = self.create_parameter((1, channel // (2 * groups), 1, 1),
                                             default_initializer=nn.initializer.Assign(
                                                 paddle.zeros((1, channel // (2 * groups), 1, 1))))
        self.sbias = self.create_parameter((1, channel // (2 * groups), 1, 1),
                                           default_initializer=nn.initializer.Assign(
                                               paddle.ones((1, channel // (2 * groups), 1, 1))))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape((b, groups, -1, h, w))
        x = x.transpose([0, 2, 1, 3, 4])

        # flatten
        x = x.reshape((b, -1, h, w))

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape((b * self.groups, -1, h, w))
        x_0, x_1 = x.chunk(2, axis=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = paddle.concat([xn, xs], axis=1)
        out = out.reshape((b, -1, h, w))

        out = self.channel_shuffle(out, 2)
        return out


# 特征融合模块SFF
class SFF_Layer(nn.Layer):
    def __init__(self, F_g, F_l, F_int):
        super(SFF_Layer, self).__init__()
        # 消融实验 concat+3*3
        self.W_g = nn.Sequential(
            nn.Conv2D(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2D(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2D(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(1),
            nn.Sigmoid()
        )

        # self.upsam = up_conv(ch_in=64, ch_out=32)
        self.relu = nn.ReLU()
        self.conv1x1 = nn.Conv2D(F_l, F_g, kernel_size=3, stride=1, padding=1)
        self.sa = SA(F_g)

    def forward(self, g, x):
        x1 = self.W_x(x)  # 64-32

        g1 = self.W_g(g)  # 32-32

        x2 = x1 + g1
        ps = self.sa(x2)

        ps1 = self.relu(ps)
        ps2 = self.psi(ps1)

        fea1 = g1 * ps2
        fea2 = x2 * ps2
        fea = paddle.concat((fea1, fea2), axis=1)
        fea = self.conv1x1(fea)

        return fea


# MA
class MA_block(nn.Layer):
    def __init__(self, ch_in, ch_out, ch_out1, height, weight):
        super(MA_block, self).__init__()
        self.ma_block1 = nn.Sequential(
            nn.Conv2D(ch_in, ch_out1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(ch_out1),
            nn.ReLU(),
            nn.Conv2D(ch_out1, ch_out, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.ma_block2 = nn.Sequential(
            nn.Conv2D(ch_in, ch_out1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(ch_out1),
            nn.ReLU(),
            nn.Conv2D(ch_out1, ch_out, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.ma_block3 = nn.Sequential(
            nn.Conv2D(ch_in, ch_out1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(ch_out1),
            nn.ReLU(),
            nn.Conv2D(ch_out1, ch_out, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.pool1 = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.pool2 = nn.AvgPool2D(kernel_size=(height, 1))
        self.pool3 = nn.AvgPool2D(kernel_size=(1, weight))

    def forward(self, x):
        y = self.pool1(x)
        y = self.ma_block1(y)
        y = x * y
        w = self.pool2(x)
        w = self.ma_block2(w)
        w = x * w
        z = self.pool3(x)
        z = self.ma_block3(z)
        z = x * z
        out = z + y + w
        return out


# 解码基本模块

# 通道数减半+上采样 用tub

class Up_conv(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(Up_conv, self).__init__()
        self.conv = nn.Conv2DTranspose(ch_in, ch_out, kernel_size=2, stride=2, padding=0)
        self.residual = nn.Conv2D(ch_in + ch_out, ch_out, kernel_size=1)
        self.up_1 = nn.UpsamplingBilinear2D(scale_factor=2)
        self.norm1 = nn.BatchNorm2D(ch_out)
        self.norm2 = nn.BatchNorm2D(ch_in)
        self.act = nn.Silu()

    def forward(self, x):
        y = self.conv(x)
        x1 = self.norm1(y)
        z = self.up_1(x)
        x2 = self.norm2(z)
        x3 = paddle.concat(x=[x1, x2], axis=1)
        x4 = self.residual(x3)
        x = self.act(x4)

        return x


# attention gate
class Att_Gate(nn.Layer):
    def __init__(self, F_g, F_l, F_int):
        super(Att_Gate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2D(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2D(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2D(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

#DRF模块
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddle.vision.transforms import transforms as T

#三个并行膨胀卷积组成MRF1
class MRF_1(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(MRF_1, self).__init__()
        self.atrous_block1=nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=3,stride=1,padding=1,dilation=1),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

        self.atrous_block2=nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=3,stride=1,padding=2,dilation=2),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

        self.atrous_block3=nn.Sequential(
            nn.Conv2D(ch_in,ch_out,kernel_size=3,stride=1,padding=4,dilation=4),
            nn.BatchNorm(ch_out),
            nn.ReLU()
        )

        self.conv=nn.Conv2D(ch_out*3, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        x1=self.atrous_block1(x)
        x2=self.atrous_block2(x)
        x3=self.atrous_block3(x)
        x=paddle.concat(x=[x1,x2,x3],axis=1)
        x=self.conv(x)
        return x

class DRF_block(nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(DRF_block, self).__init__()
        self.mrf1=MRF_1(512,512)
        self.mrf2=MRF_1(512,512)
        self.mrf3=MRF_1(512,512)
        self.conv1=nn.Conv2D(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        x1=self.mrf1(x)
        x2=x+x1
        x3=self.mrf2(x2)
        x4=x2+x3
        x5=self.mrf3(x4)
        x6=self.conv1(x)
        out=x4+x5+x6

        return out


class CFHA_Block(nn.Layer):
    def __init__(self, in_channel=3, out_channel=1):
        super(CFHA_Block, self).__init__()

        # 编码器
        self.encoder1 = conv_3x3(ch_in=in_channel, ch_out=32)
        self.down1 = TDB_Block(in_channel=32, out_channel=32, output_size=160)

        self.encoder2 = conv_block(ch_in=32, ch_out=64)
        self.down2 = TDB_Block(in_channel=64, out_channel=64, output_size=80)

        self.encoder3 = conv_block(ch_in=64, ch_out=128)
        self.down3 = TDB_Block(in_channel=128, out_channel=128, output_size=40)

        self.encoder4 = conv_block(ch_in=128, ch_out=256)
        self.down4 = TDB_Block(in_channel=256, out_channel=256, output_size=20)

        self.encoder5 = conv_block(ch_in=256, ch_out=512)

        # TUB
        self.tub1 = TUB_Block(in_channel=64, out_channel=32)
        self.tub2 = TUB_Block(in_channel=128, out_channel=64)
        self.tub3 = TUB_Block(in_channel=256, out_channel=128)
        self.tub4 = TUB_Block(in_channel=512, out_channel=256)

        self.sff1 = SFF_Layer(32, 64, 32)
        self.sff2 = SFF_Layer(64, 128, 64)
        self.sff3 = SFF_Layer(128, 256, 128)
        self.sff4 = SFF_Layer(256, 512, 256)

        # self.sff1= conv_3x3(96,32)
        # self.sff2= conv_3x3(192,64)
        # self.sff3= conv_3x3(384,128)
        # self.sff4= conv_3x3(768,256)

        # MA块
        self.ma1 = MA_block(ch_in=32, ch_out=32, ch_out1=8, height=320, weight=320)
        self.ma2 = MA_block(ch_in=64, ch_out=64, ch_out1=16, height=160, weight=160)
        self.ma3 = MA_block(ch_in=128, ch_out=128, ch_out1=32, height=80, weight=80)
        self.ma4 = MA_block(ch_in=256, ch_out=256, ch_out1=64, height=40, weight=40)

        # DRF
        self.drf = DRF_block(ch_in=512, ch_out=512)

        # 解码器
        self.Upsam4 = Up_conv(ch_in=512, ch_out=256)
        self.Att4 = Att_Gate(F_g=256, F_l=256, F_int=128)
        self.Upconv4 = conv_block(ch_in=512, ch_out=256)

        self.Upsam3 = Up_conv(ch_in=256, ch_out=128)
        self.Att3 = Att_Gate(F_g=128, F_l=128, F_int=64)
        self.Upconv3 = conv_block(ch_in=256, ch_out=128)

        self.Upsam2 = Up_conv(ch_in=128, ch_out=64)
        self.Att2 = Att_Gate(F_g=64, F_l=64, F_int=32)
        self.Upconv2 = conv_block(ch_in=128, ch_out=64)

        self.Upsam1 = Up_conv(ch_in=64, ch_out=32)
        self.Att1 = Att_Gate(F_g=32, F_l=32, F_int=16)
        self.Upconv1 = conv_block(ch_in=64, ch_out=32)

        self.conv1 = nn.Conv2D(32, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.encoder1(x)

        x1 = self.down1(x)
        x2 = self.encoder2(x1)

        x3 = self.tub1(x2)
        out1 = self.sff1(x, x3)
        y1 = self.ma1(out1)
        x4 = self.down2(x2)
        x5 = self.encoder3(x4)
        x6 = self.tub2(x5)
        out2 = self.sff2(x2, x6)

        y2 = self.ma2(out2)

        x7 = self.down3(x5)
        x8 = self.encoder4(x7)
        x9 = self.tub3(x8)
        out3 = self.sff3(x5, x9)

        y3 = self.ma3(out3)

        x10 = self.down4(x8)
        x11 = self.encoder5(x10)
        x12 = self.tub4(x11)
        out4 = self.sff4(x8, x12)

        y4 = self.ma4(out4)

        y5 = self.drf(x11)

        z1 = self.Upsam4(y5)
        z2 = self.Att4(g=z1, x=y4)
        z1 = paddle.concat(x=[z2, z1], axis=1)
        z1 = self.Upconv4(z1)

        z3 = self.Upsam3(z1)
        z4 = self.Att3(g=z3, x=y3)
        z3 = paddle.concat(x=[z4, z3], axis=1)
        z3 = self.Upconv3(z3)

        z5 = self.Upsam2(z3)
        z6 = self.Att2(g=z5, x=y2)
        z5 = paddle.concat(x=[z6, z5], axis=1)
        z5 = self.Upconv2(z5)

        z7 = self.Upsam1(z5)
        z8 = self.Att1(g=z7, x=y1)
        z7 = paddle.concat(x=[z8, z7], axis=1)
        z7 = self.Upconv1(z7)
        z8 = self.conv1(z7)

        return z8