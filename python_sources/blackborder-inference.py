#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


## REMEMBER TO CHANGE ENSEMBLE PART (SOMETIMES ONLY GET RESULT OF 1 MODEL INSTEAD OF AVERAGING)
## That's in function predict_on_video()

# Version 1: Mobilenet underbalance (with border): LB 0.437
# Version 2: EffB0 underbalance (with border): LB 0.450 
# Version 3: Mobilenet all data (with border): LB 0.381
# Version 4: blend of version 3 (with border) and EffB0 LB0.388 (version 7 of EffB9 kernel, no border): LB ????
# Version 5: Mobilenet all data (with border) gradual freeze: LB 0.42
# Version 6: Mobilenet all data (sqrimg, no border): LB 0.37
# Version 7: Blend 3 models (version 4 and version 6): LB 0.33
# Version 10: Only Xception (square img size 299): LB 0.42
# Version 11: Blend 4 models
# Version 12: NASNet (blackborder): LB 0.42
# Version 13: Blend 6 models (with last new model EffB1 square)
# Version 14: LSTM model
# Version 15: new Xception model
# Version 16: new LSTM audio model


# In[ ]:


TEST_DIR = "/kaggle/input/deepfake-detection-challenge/test_videos/"
CHECKPOINT = '/kaggle/input/kha-deepfake-dataset/checkpoint_mobilev3_alldata_1903_withfaceforensics_3epochs_.pth' # blackborder, all data
CHECKPOINT2 = '/kaggle/input/kha-deepfake-dataset/cpt_mbn_sqrimg_2503)2epochs_.pth' # square, all data
CHECKPOINT3 = '/kaggle/input/kha-deepfake-dataset/checkpoint_b0_1803_0epochs_0.4498354019969702.pth' # square, less data (only old DFDC)
CHECKPOINT4 = '/kaggle/input/kha-deepfake-dataset/cpt_xception_new_sqrimg_29030_epochs_3_moment.pth' # square, all data
CHECKPOINT5 = '/kaggle/input/kha-deepfake-dataset/cpt_nasnet_bbd_29032_epochs_0_moment.pth' # black border
CHECKPOINT6 = '/kaggle/input/kha-deepfake-dataset/cpt_effb1_sqrimg_25032_epochs_1_moment.pth' # black border
CHECKPOINT7 = '/kaggle/input/kha-deepfake-dataset/cpt_mbn_LSTM16_300313epochs_0.21526583118569945.pth' # square, old dfdc, LSTM
CHECKPOINT8 = '/kaggle/input/kha-deepfake-dataset/checkpoint_b0_LSTM16_2503_batchfirst1epochs_0.25805166250713446.pth' # square, old dfdc, LSTM with audio
CHECKPOINT9 = '/kaggle/input/kha-deepfake-dataset/cpt_nasnetnew_bbd_31031_epochs_0_moment.pth' # square, all data
CHECKPOINT10 = '/kaggle/input/kha-deepfake-dataset/cpt_xception_sqrimg_25031_epochs_0_moment.pth' # square, all data, old xception
CHECKPOINT11 = '/kaggle/input/kha-deepfake-dataset/cpt_resnet_bbd_31033_epochs_3_moment.pth'  # bbd, resnet

CONVERT_RGB = True

MTCNN_BATCH = 15
GAP = 5
IMG_SIZE = 224
SCALE = 0.5
import multiprocessing as mp
NUM_WORKERS = mp.cpu_count()


# In[ ]:





# In[ ]:


import sys
package_path = '../input/kha-efficientnet/EfficientNet-PyTorch/'
sys.path.append(package_path)
from efficientnet_pytorch import EfficientNet


# In[ ]:


get_ipython().run_cell_magic('capture', '', '# Install facenet-pytorch (with internet use "pip install facenet-pytorch")\n!pip install /kaggle/input/khafacenet/facenet_pytorch-2.2.7-py3-none-any.whl\n!pip install /kaggle/input/imutils/imutils-0.5.3')


# In[ ]:


# %%capture
# # Install facenet-pytorch (with internet use "pip install facenet-pytorch")
# !pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.2.7-py3-none-any.whl
# !pip install /kaggle/input/imutils/imutils-0.5.3


# In[ ]:


# ! tar xvf ../input/ffmpeg-static-build/ffmpeg-git-amd64-static.tar.xz


# In[ ]:


import os, sys, random, ast, numpy as np, pandas as pd, cv2, random
import torch, torchvision.models, torch.nn as nn, torch.nn.functional as F
from torchvision.transforms import Normalize
from tqdm.notebook import tqdm
import torchvision.models as models
from facenet_pytorch import MTCNN
from IPython.display import clear_output
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import torch.nn as nn

# from pathlib import Path
# import librosa
# from scipy.io import wavfile
# import subprocess

# from sklearn.externals import joblib
# scaler = joblib.load("/kaggle/input/kha-deepfake-dataset/audio_scaler")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def audio_feat_extract(aufile):
#     rate, wave = wavfile.read(aufile)
#     wave = wave[:,0].astype(float)
#     feats = []
#     S = np.abs(librosa.stft(wave))
#     feats.append(librosa.feature.poly_features(S=S, order=3))
#     feats.append(librosa.feature.mfcc(S=S, sr=rate)[[0, 1, 3, 7, 15]])
#     feats.append(librosa.feature.spectral_bandwidth(S=S, sr=rate))
#     feats.append(librosa.feature.zero_crossing_rate(wave))
#     feats.append(librosa.onset.onset_strength(S=S, sr=rate))
#     feats.append(librosa.feature.chroma_stft(S=S, sr=rate))
#     feats.append(librosa.feature.rms(S=S))
#     feats.append(librosa.feature.tonnetz(y=wave, sr=rate))
#     final_feats = []
#     for f in feats:
#         m = f.mean(axis=1) if f.ndim==2 else [f.mean()]
#         v = f.var(axis=1) if f.ndim==2 else [f.var()]
#         final_feats.extend(m)
#         final_feats.extend(v)
#     if len(final_feats) != 62: final_feats = np.zeros(62)
#     final_feats = scaler.transform([final_feats])[0]
#     final_feats = torch.tensor(final_feats)
#     return final_feats

# au_dir = '/kaggle/working/audio_files'
# os.makedirs(au_dir, exist_ok=True)

# def create_wav(file, output_dir=au_dir, aufile='lalala'):
#     command = f"../working/ffmpeg-git-20191209-amd64-static/ffmpeg -i {file} -ab 192000 -ac 2 -ar 44100 -vn {output_dir}/{aufile}.wav"
#     subprocess.call(command, shell=True)
    
# def get_au_feat(testvid):
#     aufile = testvid[:-4]
#     create_wav(TEST_DIR + testvid, aufile=aufile)
# #     print(os.listdir(au_dir))
#     fullaufile = au_dir + '/' + aufile + '.wav'
#     au_feat = audio_feat_extract(fullaufile)
#     os.remove(fullaufile)
# #     print(os.listdir(au_dir))
#     return au_feat


# In[ ]:





# In[ ]:


# MOBILENET V3

def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', 2],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', 2],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', 2],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', 2],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', 2],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', 2],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load(CFG.pretrained)
        model.load_state_dict(state_dict, strict=True)
        # raise NotImplementedError
    return model


# In[ ]:


net = mobilenetv3(mode='small', pretrained=False)
net.classifier[1] = torch.nn.Linear(in_features=1280, out_features=1)
net = net.to(device)

state_dict = torch.load(CHECKPOINT)
net.load_state_dict(state_dict)
net.cuda()
net.eval()


# In[ ]:


net2 = mobilenetv3(mode='small', pretrained=False)
net2.classifier[1] = torch.nn.Linear(in_features=1280, out_features=1)
net2 = net2.to(device)

state_dict = torch.load(CHECKPOINT2)
net2.load_state_dict(state_dict)
net2.cuda()
net2.eval()


# In[ ]:





# In[ ]:


# EFFFICIENTNET

net3 = EfficientNet.from_name("efficientnet-b0")
net3._fc = torch.nn.Linear(in_features=net3._fc.in_features, out_features=1)
net3.load_state_dict(torch.load(CHECKPOINT3))
net3.cuda()
net3.eval()


# In[ ]:


# EFFFICIENTNET

net6 = EfficientNet.from_name("efficientnet-b1")
net6._fc = torch.nn.Linear(in_features=net6._fc.in_features, out_features=1)
net6.load_state_dict(torch.load(CHECKPOINT6))
net6.cuda()
net6.eval()


# In[ ]:





# In[ ]:


# XCEPTION

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'],             "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model

net4 = Xception()
#net.load_state_dict(torch.load('F:/pretrained/xception-43020ad28.pth', map_location=torch.device(device)))
net4.last_linear = net4.fc
del net4.fc
net4.last_linear = nn.Linear(in_features=2048, out_features=1, bias=True)
net4.load_state_dict(torch.load(CHECKPOINT4))
net4 = net4.to(device)
net4.cuda()
net4.eval()


net10 = Xception()
#net.load_state_dict(torch.load('F:/pretrained/xception-43020ad28.pth', map_location=torch.device(device)))
net10.last_linear = net10.fc
del net10.fc
net10.last_linear = nn.Linear(in_features=2048, out_features=1, bias=True)
net10.load_state_dict(torch.load(CHECKPOINT10))
net10 = net10.to(device)
net10.cuda()
net10.eval()


# In[ ]:





# In[ ]:


# NASNET MOBILE
class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding, count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, name=None, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.name = name

    def forward(self, x):
        x = self.relu(x)
        if self.name == 'specific':
            x = nn.ZeroPad2d((1, 0, 1, 0))(x)
        x = self.separable_1(x)
        if self.name == 'specific':
            x = x[:, :, 1:, 1:].contiguous()

        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesReduction(BranchSeparables):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, z_padding=1, bias=False):
        BranchSeparables.__init__(self, in_channels, out_channels, kernel_size, stride, padding, bias)
        self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))

    def forward(self, x):
        x = self.relu(x)
        x = self.padding(x)
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):
    def __init__(self, stem_filters, num_filters=42):
        super(CellStem0, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.num_filters, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(2*self.num_filters, self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True))

        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters//2, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters//2, 1, stride=1, bias=False))

        self.final_path_bn = nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True)

        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(self.num_filters, self.num_filters, 7, 2, 3, name='specific', bias=False)

        # self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(self.num_filters, self.num_filters, 7, 2, 3, name='specific', bias=False)

        # self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2, name='specific', bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.num_filters, 3, 1, 1, name='specific', bias=False)
        # self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)

        x_relu = self.relu(x_conv0)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        # final path
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))

        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=0.001, momentum=0.1, affine=True)

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        # final path
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)

        self.comb_iter_1_left = BranchSeparables(out_channels_left, out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, name='specific', bias=False)

        # self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, name='specific', bias=False)

        # self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, name='specific', bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, name='specific', bias=False)
        # self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_4_right =MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NASNetAMobile(nn.Module):
    """NASNetAMobile (4 @ 1056) """

    def __init__(self, num_classes=1000, stem_filters=32, penultimate_filters=1056, filters_multiplier=2):
        super(NASNetAMobile, self).__init__()
        self.num_classes = num_classes
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier

        filters = self.penultimate_filters // 24
        # 24 is default value for the architecture

        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels=self.stem_filters, kernel_size=3, padding=0, stride=2,
                                                bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=0.001, momentum=0.1, affine=True))

        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters // (filters_multiplier ** 2))
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters // filters_multiplier)

        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left=filters//2, # 1, 0.5
                                in_channels_right=2*filters, out_channels_right=filters) # 2, 1
        self.cell_1 = NormalCell(in_channels_left=2*filters, out_channels_left=filters, # 2, 1
                                 in_channels_right=6*filters, out_channels_right=filters) # 6, 1
        self.cell_2 = NormalCell(in_channels_left=6*filters, out_channels_left=filters, # 6, 1
                                 in_channels_right=6*filters, out_channels_right=filters) # 6, 1
        self.cell_3 = NormalCell(in_channels_left=6*filters, out_channels_left=filters, # 6, 1
                                 in_channels_right=6*filters, out_channels_right=filters) # 6, 1

        self.reduction_cell_0 = ReductionCell0(in_channels_left=6*filters, out_channels_left=2*filters, # 6, 2
                                               in_channels_right=6*filters, out_channels_right=2*filters) # 6, 2

        self.cell_6 = FirstCell(in_channels_left=6*filters, out_channels_left=filters, # 6, 1
                                in_channels_right=8*filters, out_channels_right=2*filters) # 8, 2
        self.cell_7 = NormalCell(in_channels_left=8*filters, out_channels_left=2*filters, # 8, 2
                                 in_channels_right=12*filters, out_channels_right=2*filters) # 12, 2
        self.cell_8 = NormalCell(in_channels_left=12*filters, out_channels_left=2*filters, # 12, 2
                                 in_channels_right=12*filters, out_channels_right=2*filters) # 12, 2
        self.cell_9 = NormalCell(in_channels_left=12*filters, out_channels_left=2*filters, # 12, 2
                                 in_channels_right=12*filters, out_channels_right=2*filters) # 12, 2

        self.reduction_cell_1 = ReductionCell1(in_channels_left=12*filters, out_channels_left=4*filters, # 12, 4
                                               in_channels_right=12*filters, out_channels_right=4*filters) # 12, 4

        self.cell_12 = FirstCell(in_channels_left=12*filters, out_channels_left=2*filters, # 12, 2
                                 in_channels_right=16*filters, out_channels_right=4*filters) # 16, 4
        self.cell_13 = NormalCell(in_channels_left=16*filters, out_channels_left=4*filters, # 16, 4
                                  in_channels_right=24*filters, out_channels_right=4*filters) # 24, 4
        self.cell_14 = NormalCell(in_channels_left=24*filters, out_channels_left=4*filters, # 24, 4
                                  in_channels_right=24*filters, out_channels_right=4*filters) # 24, 4
        self.cell_15 = NormalCell(in_channels_left=24*filters, out_channels_left=4*filters, # 24, 4
                                  in_channels_right=24*filters, out_channels_right=4*filters) # 24, 4

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(7, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(24*filters, self.num_classes)

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)

        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)

        x_reduction_cell_0 = self.reduction_cell_0(x_cell_3, x_cell_2)

        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_3)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)

        x_reduction_cell_1 = self.reduction_cell_1(x_cell_9, x_cell_8)

        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_9)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        return x_cell_15

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x
    
net5 = NASNetAMobile()
net5.last_linear = nn.Linear(in_features=1056, out_features=1, bias=True)    
net5.load_state_dict(torch.load(CHECKPOINT5))
net5 = net5.to(device)
net5.cuda()
net5.eval()


net9 = NASNetAMobile()
net9.last_linear = nn.Linear(in_features=1056, out_features=1, bias=True)    
net9.load_state_dict(torch.load(CHECKPOINT9))
net9 = net9.to(device)
net9.cuda()
net9.eval()


# In[ ]:





# In[ ]:


class CFG:
    seq_len=10
    lstm_in = 16
    lstm_out = 16

class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.cnn_net = mobilenetv3(mode='small', pretrained=False)
        self.cnn_net.classifier[1] = nn.Linear(in_features=1280, out_features=1)  
        self.cnn_net.classifier[1] = nn.Linear(in_features=1280, out_features=CFG.lstm_in) 
        self.lstm = nn.LSTM(CFG.lstm_in, CFG.lstm_out, bidirectional=False, batch_first=True)
        self.reg_layer = nn.Sequential(nn.Dropout(0.5),
                                       nn.Linear(CFG.lstm_out, 1) )        
        
    def forward(self, x):  # x: n_samples x seq_len x 3 x 224 x 224     
        n_samples = x.shape[0]
        x = x.view(-1, 3, 224, 224)
        x = self.cnn_net(x) # shape now: (n_samples x seq_len) x CFG.final_layer_size = (3x15)x24 = 45x24
        x = x.view(n_samples, CFG.seq_len, -1)
        x, (h_n, h_c) = self.lstm(x)   # x = seq_len, batch, num_directions*hidden_size  
        x = x[:,-1]  # Take last time step, which will be BATCH_SIZE * (2*HIDDEN) -> e.g. 12 * 128
        y = self.reg_layer(x)
        return y    
    
net7 = LSTM_Model()
net7.load_state_dict(torch.load(CHECKPOINT7))
net7 = net7.to(device)
net7.cuda()
net7.eval()

net8 = LSTM_Model()
net8.load_state_dict(torch.load(CHECKPOINT8))
net8 = net8.to(device)
net8.cuda()
net8.eval()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# class Audio_LSTM_Model(nn.Module):
#     def __init__(self):
#         super(Audio_LSTM_Model, self).__init__()
#         self.cnn_net = mobilenetv3(mode='small', pretrained=False)
#         self.cnn_net.classifier[1] = nn.Linear(in_features=1280, out_features=1)  
#         self.cnn_net.classifier[1] = nn.Linear(in_features=1280, out_features=CFG.lstm_in) 
#         self.lstm = nn.LSTM(CFG.lstm_in, CFG.lstm_out, bidirectional=False, batch_first=True)
#         self.reg_layer = nn.Sequential(nn.Dropout(0.5),
#                                        nn.Linear(CFG.lstm_out+62, 1) )        
        
#     def forward(self, x, au):  # x: n_samples x seq_len x 3 x 224 x 224
# #         print('aushape',au.shape)
#         n_samples = x.shape[0]
#         x = x.view(-1, 3, 224, 224)
# #         print('x0',x.shape)
#         x = self.cnn_net(x) # shape now: (n_samples x seq_len) x CFG.final_layer_size = (3x15)x24 = 45x24
# #         print('x1',x.shape)
#         x = x.view(n_samples, CFG.seq_len, -1)
# #         print('x2',x.shape)
#         x, (h_n, h_c) = self.lstm(x)   # x = seq_len, batch, num_directions*hidden_size
# #         print('x3',x.shape)
#         x = x[:,-1]  # Take last time step, which will be BATCH_SIZE * (2*HIDDEN) -> e.g. 12 * 128
# #         print('x4',x.shape) # 32 x 16
#         x = torch.cat((x, au), 1)
# #         print('x5',x.shape)
#         y = self.reg_layer(x)
# #         print('x6',y.shape)
#         return y    

# net8 = Audio_LSTM_Model()
# net8.load_state_dict(torch.load(CHECKPOINT8))
# net8 = net8.to(device)
# net8.cuda()
# net8.eval()


# In[ ]:





# In[ ]:


net11 = torchvision.models.resnet18(pretrained=False)
net11.fc = nn.Linear(in_features=512, out_features=1, bias=True)
net11.load_state_dict(torch.load(CHECKPOINT11))
net11 = net11.to(device)
net11.cuda()
net11.eval()


# In[ ]:





# In[ ]:


import albumentations
from albumentations.augmentations.transforms import ShiftScaleRotate, HorizontalFlip, RandomBrightnessContrast, MotionBlur, Blur, GaussNoise, JpegCompression


# In[ ]:


def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = (size - h)//2
    b = (size - h) - (size - h)//2
    l = (size - w)//2
    r = (size - w) - (size-w)//2
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)


def correct_box_of_one_video(boxes_of_one_video, probs_of_one_video):
    B = boxes_of_one_video
    P = probs_of_one_video
    recent_cluster = []
    new_B = []

    for i, (boxes, probs) in enumerate(zip(B, P)):
        if len(boxes) == 0: 
            new_B.append([])
            continue
            
        boxes, probs = np.array(boxes), np.array(probs)
        boxes = boxes[probs>0.95].tolist()
        probs = probs[probs>0.95].tolist()
        
        if len(boxes) == 0: 
            new_B.append([])
            continue
        
        if len(boxes)!=len(probs): 
            new_B.append([])
            continue

        boxes = [[item if item>=0 else 0 for item in box] for box in boxes]
        
        if len(recent_cluster) == 0: # if the first frame with multiple faces, select the best one
            best = np.argmax(probs)
            recent_cluster = boxes[best] 
            new_B.append(boxes[best])
            continue

        else:
            best_overlap = 0
            best_m = -1

            for m, b in enumerate(boxes):
                iou = IoU(b, recent_cluster)
                if iou > 0.2: 
                    if iou > best_overlap:
                        best_m = m
                        best_overlap = iou
                        
            if best_m != -1: 
                new_B.append(boxes[best_m])
                recent_cluster = boxes[best_m]
            else: 
                new_B.append([])
    final_boxes, start_, end_ = rectify_boxes(new_B)
    return final_boxes, start_, end_             


def rectify_boxes(boxes):
    
    start_frame = None
    for i, b in enumerate(boxes):
        # Determine proper video start
        if len(b)!=0:
            start_frame = i
            break

    end_frame = None
    for i, b in enumerate(reversed(boxes)):
        # Determine proper video end
        if len(b)!=0:
            end_frame = len(boxes)-i-1
            break    

    if start_frame is None or end_frame is None or start_frame==end_frame: 
        boxes = None            
    else:
        boxes = boxes[start_frame:end_frame+1]

        for idx in range(len(boxes)):
            box = boxes[idx]
            if len(box)!=0: continue

            idx_next_final = -1
            idx_prev_final = -1

            for idx_next in range(idx+1, len(boxes)): # search for next idx
                if idx_next < len(boxes):
                    if len(boxes[idx_next])!=0: 
                        idx_next_final = idx_next
                        break

            for idx_prev in reversed(range(idx)): # search for prev idx
                if idx_prev >= 0:
                    if  len(boxes[idx_prev])!=0: 
                        idx_prev_final = idx_prev
                        break

            if idx_next_final != -1 and idx_prev_final == -1: 
                boxes[idx] = boxes[idx_next_final]

            elif idx_next_final == -1 and idx_prev_final != -1: 
                boxes[idx] = boxes[idx_prev_final]

            elif idx_next_final != -1 and idx_prev_final != -1: 
                boxes[idx] = regress_box(boxes, idx, idx_prev_final, idx_next_final)

            elif idx_next_final == -1 and idx_prev_final == -1: pass

    return boxes, start_frame, end_frame

    
def regress_box(b, i, prev_i, next_i):
    w1, w2 = abs(i-prev_i), abs(i-next_i)
    new_box = (np.array(b[prev_i])*w2 + np.array(b[next_i])*w1 ) / (w1+w2)
    new_box = new_box.astype(int).tolist()
    return new_box


def IoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def incre_counter(counter, idx):
    if idx not in counter.keys(): counter[idx] = 1
    else: counter[idx] += 1
    return counter

def get_len_cluster(b):
    return sum([len(item) for item in b])

MAX_FRAMES = 30

augment_ = albumentations.Compose([ShiftScaleRotate(p=0.2, scale_limit=0.1, border_mode=1, rotate_limit=10), # bad
                                        RandomBrightnessContrast(p=0.25, brightness_limit=0.1, contrast_limit=0.1),
                                        GaussNoise(p=.1),
                                        JpegCompression(p=.1, quality_lower=50)])

def get_augment(img):
    res = augment_(image=img)
    img = res['image']
    return img

def extract_frames(filepath):
    v_cap = cv2.VideoCapture(filepath)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = 1
    this_video_boxes = []
    this_video_probs = []
    
    # Get frames and boxes
    frames = []
    frames_RGB = []
    count_frame_for_mtcnn_batch = 0
    if v_len > 0:
        frame_skip = int(np.round(v_len/MAX_FRAMES))
#         print('vlen', v_len, 'frame_skip', frame_skip)
        for j in range(v_len):
            ret = v_cap.grab()
            if j % frame_skip != 0: continue

            if not ret: continue

            ret, frame = v_cap.retrieve()
            if not ret or frame is None: continue
            
            if CONVERT_RGB: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_RGB.append(frame)
            frame = Image.fromarray(frame)
            frames.append(frame)
            count_frame_for_mtcnn_batch += 1
            
            if count_frame_for_mtcnn_batch == MTCNN_BATCH or j == v_len - 1:
                boxes, probs = fast_mtcnn(frames)
                this_video_boxes.extend(boxes)
                this_video_probs.extend(probs)
                count_frame_for_mtcnn_batch = 0
                frames = []

    v_cap.release()

    boxes, start_, end_ = correct_box_of_one_video(this_video_boxes, this_video_probs)
    
    # Final step: build faces
    if boxes is not None:
        frames_RGB = frames_RGB[start_:end_+1]
        for j, (frame, box) in enumerate(zip(frames_RGB, boxes)):
            
            ### Box1: with blackborder padding
            box = [int(item/SCALE) for item in box]
            face_ori = frame[box[1]:box[3], box[0]:box[2], :]
            face_ori = make_square_image(isotropically_resize_image(face_ori, IMG_SIZE))
            face_ori = get_augment(face_ori)
            

            ########
    
            ### Box2: no blackborder
            box2 = box
            dim1, dim2 = abs(box2[1]-box2[3]), abs(box2[0]-box2[2])
            gap = abs(dim1 - dim2)
            if dim1 < dim2:
                box2[1] -= gap//2
                box2[3] += gap//2
            elif dim1 > dim2:
                box2[0] -= gap//2
                box2[2] += gap//2                  
            box2 = [item if item >= 0 else 0 for item in box2]
            face_ori_sqr = frame[box2[1]:box2[3], box2[0]:box2[2], :]
            face_ori_sqr = cv2.resize(face_ori_sqr, (IMG_SIZE, IMG_SIZE))
            face_lstm_ori = face_ori_sqr.copy()
            face_ori_sqr = get_augment(face_ori_sqr)
            face_ori_299_sqr = cv2.resize(face_ori_sqr, (299, 299))
            face_ori_299_sqr = get_augment(face_ori_299_sqr)
            #####
            
            ### Make batch imgs for frame models
            for k in range(2):
                face = face_ori
                face_sqr = face_ori_sqr
                face_299_sqr = face_ori_299_sqr
                face_lstm = face_lstm_ori
                
                if k==1: 
                    face = cv2.flip(face, 1)
                    face_sqr = cv2.flip(face_sqr, 1)
                    face_299_sqr = cv2.flip(face_299_sqr, 1)
                    face_lstm = cv2.flip(face_lstm, 1)

#                 plt.figure()
#                 plt.imshow(face_299_sqr)
                
                face = torch.tensor(face).permute((2, 0, 1)).float().div(255.)
                face_sqr = torch.tensor(face_sqr).permute((2, 0, 1)).float().div(255.)
                face_299_sqr = torch.tensor(face_299_sqr).permute((2, 0, 1)).float().div(255.)
                face_lstm = torch.tensor(face_lstm).permute((2, 0, 1)).float().div(255.)
                
                normalize_transform = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                face = normalize_transform(face)
                face_sqr = normalize_transform(face_sqr)
                face_299_sqr = normalize_transform(face_299_sqr)
                face_lstm = normalize_transform(face_lstm)

                if (j == 0) and (k==0): 
                    faces = face.unsqueeze(0)
                    faces_sqr = face_sqr.unsqueeze(0)
                    faces_299_sqr = face_299_sqr.unsqueeze(0)
                    faces_lstm = face_lstm.unsqueeze(0)
                else: 
                    faces = torch.cat((faces, face.unsqueeze(0)), 0)
                    faces_sqr = torch.cat((faces_sqr, face_sqr.unsqueeze(0)), 0)
                    faces_299_sqr = torch.cat((faces_299_sqr, face_299_sqr.unsqueeze(0)), 0)
                    faces_lstm = torch.cat((faces_lstm, face_lstm.unsqueeze(0)), 0)
    
    else: faces, face_sqr, faces_299_sqr, faces_lstm = None, None, None, None
    return faces, faces_sqr, faces_299_sqr, faces_lstm, frame_skip


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# CHECKPOINT = '/kaggle/input/kha-deepfake-dataset/checkpoint_mobilev3_alldata_1903_withfaceforensics_3epochs_.pth' # blackborder, all data
# CHECKPOINT2 = '/kaggle/input/kha-deepfake-dataset/cpt_mbn_sqrimg_2503)2epochs_.pth' # square, all data
# CHECKPOINT3 = '/kaggle/input/kha-deepfake-dataset/checkpoint_b0_1803_0epochs_0.4498354019969702.pth' # square, less data (only old DFDC)
# CHECKPOINT4 = '/kaggle/input/kha-deepfake-dataset/cpt_xception_sqrimg_25031_epochs_0_moment.pth' # square, all data
# CHECKPOINT5 = '/kaggle/input/kha-deepfake-dataset/cpt_nasnet_bbd_29032_epochs_0_moment.pth' # black border
# CHECKPOINT6 = '/kaggle/input/kha-deepfake-dataset/cpt_effb1_sqrimg_25032_epochs_1_moment.pth' # black border

def predict_on_video(model, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, video_path):
#     try:
    x, x_sqr, x_299_sqr, x_lstm, frame_skip = extract_frames(video_path)
    if x is None or x_sqr is None: return 0.5
    else:
        with torch.no_grad():
            y_pred = model(x.to(device))
            y_pred = torch.sigmoid(y_pred.squeeze().mean())

            y_pred2 = model2(x_sqr.to(device))
            y_pred2 = torch.sigmoid(y_pred2.squeeze().mean())

            y_pred3 = model3(x_sqr.to(device))
            y_pred3 = torch.sigmoid(y_pred3.squeeze().mean())

            y_pred4 = model4(x_299_sqr.to(device))
            y_pred4 = torch.sigmoid(y_pred4.squeeze().mean())

            y_pred5 = model5(x.to(device))
            y_pred5 = torch.sigmoid(y_pred5.squeeze().mean())

            y_pred6 = model6(x_sqr.to(device))
            y_pred6 = torch.sigmoid(y_pred6.squeeze().mean())

            ########################

            ## Create batch of sequences for LSTM model (net7)
            seq_len = len(x_lstm)//2
            x_sqr_1, x_sqr_2 = x_lstm[:seq_len],  x_lstm[seq_len:] # 1 is original imgs, 2 is fliped imgs
#                 print('s1',x_sqr_1.shape)
            skip = max(1, seq_len//15) # get about 15 sequences for each of (original imgs, flipped imgs)
            start_indices = list(range(0, seq_len, skip))
#                 print('s2',start_indices)

            gap = max(1, int(np.round(20/frame_skip))) # result of training setting (training: get each 4 frames from original vid, then skip 5 for sequence generation (total gap = 20))
#                 print('gap', gap)
            for ss, start in enumerate(start_indices):
                indices = list(range(start, 500, gap))
                indices = [i%seq_len for i in indices]
                indices = indices[:CFG.seq_len]
#                     print('indices', indices)

                seqs1, seqs2 = x_sqr_1[indices].unsqueeze(0) , x_sqr_2[indices].unsqueeze(0) 
                if ss == 0: 
                    batch_seqs_1 = seqs1
                    batch_seqs_2 = seqs2
                else: 
                    batch_seqs_1 = torch.cat((batch_seqs_1, seqs1), 0)
                    batch_seqs_2 = torch.cat((batch_seqs_2, seqs2), 0)

            batch_seqs = torch.cat((batch_seqs_1, batch_seqs_2), 0)

#                 print('batch_seqs shape', batch_seqs.shape)
            y_pred7 = model7(batch_seqs.to(device))
            y_pred7 = torch.sigmoid(y_pred7.squeeze().mean())     

            y_pred8 = model8(batch_seqs.to(device))
            y_pred8 = torch.sigmoid(y_pred8.squeeze().mean())    
#                 ###########


#                 au_feat = get_au_feat(video_path.split('/')[-1])
#                 au_feats = torch.FloatTensor(len(batch_seqs), 62)
#                 for j in range(len(batch_seqs)): au_feats[j] = au_feat
#                 au_feats = au_feats.to(device).float()
# #                 print('aushape', au_feats.shape)                
#                 y_pred8 = model8(batch_seqs.to(device), au_feats)
#                 y_pred8 = torch.sigmoid(y_pred8.squeeze().mean())


    ########################

            y_pred9 = model9(x_sqr.to(device))
            y_pred9 = torch.sigmoid(y_pred9.squeeze().mean())        

            y_pred10 = model10(x_299_sqr.to(device))
            y_pred10 = torch.sigmoid(y_pred10.squeeze().mean())     

            y_pred11 = model11(x.to(device))
            y_pred11 = torch.sigmoid(y_pred11.squeeze().mean())  

            w = [3, 2, 8, 2, 2, 2, 4, 4, 2, 2, 3]

            return (w[0]*y_pred.item() + w[1]*y_pred2.item() + w[2]*y_pred3.item() + w[3]*y_pred4.item() 
                    + w[4]*y_pred5.item() + w[5]*y_pred6.item() + 
                    w[6]*y_pred7.item() + w[7]*y_pred8.item() + 
                    w[8]*y_pred9.item() + w[9]*y_pred10.item() + w[10]*y_pred11.item())/sum(w)

                
#     except Exception as e:
#         print("Prediction error on video", e)
#         return 0.5
    
    return 0.5


# In[ ]:





# In[ ]:


# a=predict_on_video(net, net2, net3, net4, net5, net6, net7, net8, net9, net10, net11, os.path.join(TEST_DIR, test_videos[0]))
# a


# In[ ]:





# In[ ]:


class FastMTCNN(object):
    def __init__(self, resize=1, *args, **kwargs):
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        if self.resize != 1:
            frames = [f.resize([int(d * self.resize) for d in f.size]) for f in frames]    
        boxes, probs = self.mtcnn.detect(frames)
        boxes = [b.astype(int).tolist() if b is not None and type(b)==np.ndarray else [] for b in boxes] 
        probs = [b.tolist() if b is not None and type(b)==np.ndarray else [] for b in probs] 
        return boxes, probs


fast_mtcnn = FastMTCNN(
    resize=SCALE,
    margin=14,
    keep_all=True,
    device=device
)


# In[ ]:


test_videos = sorted([x for x in os.listdir(TEST_DIR) if x[-4:] == ".mp4"])
len(test_videos)


# In[ ]:


# predictions = []
# for i in range(len(test_videos)):
#     filename = test_videos[i]
#     y_pred = predict_on_video(net, net2, net3, net4, net5, net6, net7, net8, net9, net10, net11, os.path.join(TEST_DIR, filename))
#     print(i, y_pred)
#     predictions.append(y_pred)


# In[ ]:





# In[ ]:


from concurrent.futures import ThreadPoolExecutor

def predict_on_video_set(model, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, videos, num_workers):
    def process_file(i):
        filename = videos[i]
        y_pred = predict_on_video(model, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11, os.path.join(TEST_DIR, filename))
        print(i, y_pred)
        return y_pred

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(videos)))

    return list(predictions)

predictions = predict_on_video_set(net, net2, net3, net4, net5, net6, net7, net8, net9, net10, net11, test_videos, num_workers=NUM_WORKERS)


# In[ ]:





# In[ ]:


predictions = np.clip(predictions, 0.005, 0.995)
submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})
submission_df.to_csv("submission.csv", index=False)


# In[ ]:


plt.hist(predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




