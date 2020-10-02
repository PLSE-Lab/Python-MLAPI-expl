import os
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F


#from lib.net.sync_bn.nn import BatchNorm2dSync as SynchronizedBatchNorm2d
#BatchNorm2d = SynchronizedBatchNorm2d
BatchNorm2d = nn.BatchNorm2d


IS_PYTORCH_PAD = True  # True  # False

### efficientnet #######################################################################

def drop_connect(x, probability, training):
    if not training: return x

    batch_size = len(x)
    keep_probability = 1 - probability
    noise = keep_probability
    noise += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    mask = torch.floor(noise)
    x = x / keep_probability * mask

    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Identity(nn.Module):
    def forward(self, x):
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(len(x),-1)

class Conv2dBn(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, zero_pad=[0,0,0,0], group=1):
        super(Conv2dBn, self).__init__()
        if IS_PYTORCH_PAD: zero_pad = [kernel_size//2]*4
        self.pad  = nn.ZeroPad2d(zero_pad)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=0, stride=stride, groups=group, bias=False)
        self.bn   = BatchNorm2d(out_channel, eps=1e-03, momentum=0.01)
        #print(zero_pad)

    def forward(self, x):
        x = self.pad (x)
        x = self.conv(x)
        x = self.bn  (x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, reduction_channel, excite_size):
        super(SqueezeExcite, self).__init__()
        self.excite_size=excite_size

        self.squeeze = nn.Conv2d(in_channel, reduction_channel, kernel_size=1, padding=0)
        self.excite  = nn.Conv2d(reduction_channel, in_channel, kernel_size=1, padding=0)
        self.act = Swish()

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x,1)
        s = self.act(self.squeeze(s))
        s = torch.sigmoid(self.excite(s))

        x = s*x
        return x


    # def forward(self, x):
    #
    #     s = F.avg_pool2d(x, kernel_size=self.excite_size)
    #     s = self.act(self.squeeze(s))
    #     s = torch.sigmoid(self.excite(s))
    #     s = F.interpolate(s, size=(x.shape[2],x.shape[3]), mode='nearest')
    #
    #     x = s*x
    #     return x

#---

class EfficientBlock(nn.Module):

    def __init__(self, in_channel, channel, out_channel, kernel_size, stride, zero_pad, excite_size, drop_connect_rate):
        super().__init__()
        self.is_shortcut = stride == 1 and in_channel == out_channel
        self.drop_connect_rate = drop_connect_rate

        if in_channel == channel:
            self.bottleneck = nn.Sequential(
                Conv2dBn(   channel, channel, kernel_size=kernel_size, stride=stride, zero_pad=zero_pad, group=channel),
                Swish(),
                SqueezeExcite(channel, in_channel//4, excite_size) if excite_size>0
                else Identity(),
                Conv2dBn(channel, out_channel, kernel_size=1, stride=1),
            )
        else:
            self.bottleneck = nn.Sequential(
                Conv2dBn(in_channel, channel, kernel_size=1, stride=1),
                Swish(),
                Conv2dBn(   channel, channel, kernel_size=kernel_size, stride=stride, zero_pad=zero_pad, group=channel),
                Swish(),
                SqueezeExcite(channel, in_channel//4, excite_size) if excite_size>0
                else Identity(),
                Conv2dBn(channel, out_channel, kernel_size=1, stride=1)
            )

    def forward(self, x):
        b = self.bottleneck(x)

        if self.is_shortcut:
            if self.training: b = drop_connect(b, self.drop_connect_rate, True)
            x = b + x
        else:
            x = b
        return x



# https://arogozhnikov.github.io/einops/pytorch-examples.html

# actual padding used in tensorflow
class EfficientNet(nn.Module):

    def __init__(self, drop_connect_rate=0.2):
        super(EfficientNet, self).__init__()
        d = drop_connect_rate

        # bottom-top
        self.stem  = nn.Sequential(
            Conv2dBn(3,32, kernel_size=3,stride=2,zero_pad=[0,1,0,1]),
            Swish()
        )

        self.block1 = nn.Sequential(
               EfficientBlock( 32,  32,  16, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=128, drop_connect_rate=d*1/7),
        )
        self.block2 = nn.Sequential(
               EfficientBlock( 16,  96,  24, kernel_size=3, stride=2, zero_pad=[0,1,0,1], excite_size= 64, drop_connect_rate=d*2/7),
            * [EfficientBlock( 24, 144,  24, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size= 64, drop_connect_rate=d*2/7) for i in range(1,2)],
        )
        self.block3 = nn.Sequential(
               EfficientBlock( 24, 144,  40, kernel_size=5, stride=2, zero_pad=[1,2,1,2], excite_size= 32, drop_connect_rate=d*3/7),
            * [EfficientBlock( 40, 240,  40, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 32, drop_connect_rate=d*3/7) for i in range(1,2)],
        )
        self.block4 = nn.Sequential(
               EfficientBlock( 40, 240,  80, kernel_size=3, stride=2, zero_pad=[0,1,0,1], excite_size= 16, drop_connect_rate=d*4/7),
            * [EfficientBlock( 80, 480,  80, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size= 16, drop_connect_rate=d*4/7) for i in range(1,3)],
        )
        self.block5 = nn.Sequential(
               EfficientBlock( 80, 480, 112, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 16, drop_connect_rate=d*5/7),
            * [EfficientBlock(112, 672, 112, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size= 16, drop_connect_rate=d*5/7) for i in range(1,3)],
        )
        self.block6 = nn.Sequential(
               EfficientBlock(112, 672, 192, kernel_size=5, stride=2, zero_pad=[1,2,1,2], excite_size=  8, drop_connect_rate=d*6/7),
            * [EfficientBlock(192,1152, 192, kernel_size=5, stride=1, zero_pad=[2,2,2,2], excite_size=  8, drop_connect_rate=d*6/7) for i in range(1,4)],
        )
        self.block7 = nn.Sequential(
               EfficientBlock(192,1152, 320, kernel_size=3, stride=1, zero_pad=[1,1,1,1], excite_size=  8, drop_connect_rate=d*7/7),
        )
        self.last = nn.Sequential(
            Conv2dBn(320,1280, kernel_size=1,stride=1),
            Swish()
        )

        self.logit = nn.Linear(1280,1000)

    def forward(self, x):
        batch_size = len(x)

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.last(x)

        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)

        return logit




##---
# net = EfficientNet()
# print(net)
