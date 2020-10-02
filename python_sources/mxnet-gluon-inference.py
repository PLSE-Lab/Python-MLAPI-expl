#!/usr/bin/env python
# coding: utf-8

# In[1]:


# coding = utf-8
import mxnet as mx
from mxnet.gluon import data, HybridBlock, nn
import pandas as pd
import cv2
import os
import numpy as np
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo import vision
import glob
from mxnet import nd as F, gluon
from gluoncv import model_zoo as gm


# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')


# In[ ]:


ls ../input/mxnet-gluon-baseline/model/


# In[ ]:


from gluoncv.model_zoo.resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock

class ResNetBackbone(mx.gluon.HybridBlock):
    def __init__(self, backbone='resnet50', pretrained_base=True,dilated=True, **kwargs):
        super(ResNetBackbone, self).__init__()

        with self.name_scope():
            if backbone == 'resnet50':
                pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
            elif backbone == 'resnet101':
                pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
            elif backbone == 'resnet152':
                pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
            else:
                raise RuntimeError(f'unknown backbone: {backbone}')

            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.relu = pretrained.relu
            self.maxpool = pretrained.maxpool
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4


# In[ ]:


import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock

class ResNetFPN(mx.gluon.HybridBlock):
    def __init__(self, backbone= 'resnet50', backbone_lr_mult=0.1, **kwargs):
        super(ResNetFPN, self).__init__()

        self.backbone_name = backbone
        self.backbone_lr_mult = backbone_lr_mult
        self._kwargs = kwargs

        with self.name_scope():
            self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False, dilated=False, **kwargs)

            self.head = _FPNHead(output_channels=256, **kwargs)

    def load_pretrained_weights(self):
        pretrained = ResNetBackbone(backbone=self.backbone_name, pretrained_base=True, dilated=False, **self._kwargs)
        backbone_params = self.backbone.collect_params()
        pretrained_weights = pretrained.collect_params()
        for k, v in pretrained_weights.items():
            param_name = backbone_params.prefix + k[len(pretrained_weights.prefix):]
            backbone_params[param_name].set_data(v.data())

        self.backbone.collect_params().setattr('lr_mult', self.backbone_lr_mult)

    def hybrid_forward(self,F, x):
        c1, c2, c3, c4 = self.backbone(x)
        p1, p2, p3, p4 = self.head(c1, c2, c3, c4)

        return p1, p2, p3, p4

class ResNetUnet(mx.gluon.HybridBlock):
    def __init__(self, backbone= 'resnet101', backbone_lr_mult=0.1, cls_branch=False, **kwargs):
        super(ResNetUnet, self).__init__()

        self.backbone_name = backbone
        self.backbone_lr_mult = backbone_lr_mult
        self.cls_branch = cls_branch
        self._kwargs = kwargs
        
        with self.name_scope():
            self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False, dilated=False, **kwargs)

            self.head = _UnetHead(**kwargs)

    def load_pretrained_weights(self):
        pretrained = ResNetBackbone(backbone=self.backbone_name, pretrained_base=True, dilated=False, **self._kwargs)
        backbone_params = self.backbone.collect_params()
        pretrained_weights = pretrained.collect_params()
        for k, v in pretrained_weights.items():
            param_name = backbone_params.prefix + k[len(pretrained_weights.prefix):]
            backbone_params[param_name].set_data(v.data())

        self.backbone.collect_params().setattr('lr_mult', self.backbone_lr_mult)

    def hybrid_forward(self,F, x):
        c1, c2, c3, c4 = self.backbone(x)
        out = self.head(c1, c2, c3, c4)
        if self.cls_branch:
            logits = F.max(F.mean(out, axis=1), axis=(1, 2))
            return out, logits
        return out

class _DecoderBlock(HybridBlock):
    def __init__(self, output_channels, norm_layer=nn.BatchNorm):
        super(_DecoderBlock, self).__init__()

        with self.name_scope():
            self.block = nn.HybridSequential()
            self.block.add(ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer))
            self.block.add(ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer))

    def hybrid_forward(self, F, x, y=None):

        if y is not None:
            x = F.contrib.BilinearResize2D(x, scale_height=2, scale_width=2)
            x = F.concat(x, y, dim=1)
        out = self.block(x)

        return out


class _UnetHead(HybridBlock):
    def __init__(self, num_classes, output_channels=[256, 128, 64, 32], scale=4, norm_layer=nn.BatchNorm):
        super(_UnetHead, self).__init__()
        
        self.scale = scale
        with self.name_scope():
            self.block4 = _DecoderBlock(output_channels[0], norm_layer=norm_layer)
            self.block3 = _DecoderBlock(output_channels[1], norm_layer=norm_layer)
            self.block2 = _DecoderBlock(output_channels[2], norm_layer=norm_layer)
            self.block1 = _DecoderBlock(output_channels[3], norm_layer=norm_layer)
            self.postprocess_block = nn.Conv2D(num_classes, kernel_size=1)

    def hybrid_forward(self, F, c1, c2, c3, c4):

        p4 = self.block4(c4)
        p3 = self.block3(p4, c3)
        p2 = self.block2(p3, c2)
        p1 = self.block1(p2, c1)
        if self.scale > 1:
            p1 = F.contrib.BilinearResize2D(p1, scale_height=self.scale, scale_width=self.scale)
        out = self.postprocess_block(p1)

        return out


class _FPNHead(HybridBlock):
    def __init__(self, output_channels=256, norm_layer=nn.BatchNorm):
        super(_FPNHead, self).__init__()
        self._hdsize = {}

        with self.name_scope():
            self.block4 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)
            self.block3 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)
            self.block2 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)
            self.block1 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)

    def hybrid_forward(self, F, c1, c2, c3, c4):
        p4 = self.block4(c4)
        p3 = self._resize_as(F, 'id_1', p4, c3) + self.block3(c3)
        p2 = self._resize_as(F, 'id_2', p3, c2) + self.block2(c2)
        p1 = self._resize_as(F, 'id_3', p2, c1) + self.block1(c1)

        return p1, p2, p3, p4

    def _resize_as(self, F, name, x, y):
        h_key = name + '_h'
        w_key = name + '_w'

        if hasattr(y, 'shape'):
            _, _, h, w = y.shape
            _, _, h2, w2 = x.shape

            if h == h2 and w == w2:
                h = 0
                w = 0

            self._hdsize[h_key] = h
            self._hdsize[w_key] = w
        else:
            h, w = self._hdsize[h_key], self._hdsize[w_key]

        if h == 0 and w == 0:
            return x
        else:
            return F.contrib.BilinearResize2D(x, height=h, width=w)


class SemanticFPNHead(HybridBlock):
    def __init__(self, num_classes, output_channels=128, norm_layer=nn.BatchNorm):
        super(SemanticFPNHead, self).__init__()
        self._hdsize = {}

        with self.name_scope():
            self.block4_1 = ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer)
            self.block4_2 = ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer)
            self.block4_3 = ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer)

            self.block3_1 = ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer)
            self.block3_2 = ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer)

            self.block2 = ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer)
            self.block1 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)

            self.postprocess_block = nn.Conv2D(num_classes, kernel_size=1)

    def hybrid_forward(self, F, c1, c2, c3, c4):
        out4 = self._resize_as(F, 'id_1', self.block4_1(c4), c3)
        out4 = self._resize_as(F, 'id_2', self.block4_2(out4), c2)
        out4 = self._resize_as(F, 'id_3', self.block4_3(out4), c1)

        out3 = self._resize_as(F, 'id_4', self.block3_1(c3), c2)
        out3 = self._resize_as(F, 'id_5', self.block3_2(out3), c1)

        out2 = self._resize_as(F, 'id_6', self.block2(c2), c1)

        out1 = self.block1(c1)

        out = out1 + out2 + out3 + out4

        out = self.postprocess_block(out)
        out = F.contrib.BilinearResize2D(out,scale_height=4,scale_width=4)
        return out

    def _resize_as(self, F,name, x, y):
        h_key = name + '_h'
        w_key = name + '_w'

        if hasattr(y, 'shape'):
            _, _, h, w = y.shape
            _, _, h2, w2 = x.shape

            if h == h2 and w == w2:
                h = 0
                w = 0

            self._hdsize[h_key]=h
            self._hdsize[w_key]=w
        else:
            h, w = self._hdsize[h_key], self._hdsize[w_key]

        if h == 0 and w == 0:
            return x
        else:
            return F.contrib.BilinearResize2D(x,height=h,width=w)


class ConvBlock(HybridBlock):
    def __init__(self, output_channels, kernel_size, padding=0, activation='relu', norm_layer=nn.BatchNorm):
        super().__init__()
        self.body = nn.HybridSequential()
        self.body.add(
            nn.Conv2D(output_channels, kernel_size=kernel_size, padding=padding, activation=activation),
            norm_layer(in_channels=output_channels)
        )

    def hybrid_forward(self, F, x):
        return self.body(x)


# In[ ]:


class SteelUnet(HybridBlock):
    
    def __init__(self, n_classes=5, ctx=mx.cpu()):
        super().__init__()
        with self.name_scope():
            self.feature_extractor = ResNetFPN()
            self.segment_head = SemanticFPNHead(num_classes=n_classes)
    def hybrid_forward(self, F, x):
        fpn_feature = self.feature_extractor(x)
        segment_out = self.segment_head(*fpn_feature)
        return segment_out


# In[ ]:


ctx = mx.gpu()
unet = ResNetUnet(output_channels=[256, 128, 64, 32], num_classes=5)
unet.load_parameters('../input/mxnet-gluon-baseline/unet_14_-1.params')
unet.collect_params().reset_ctx(ctx)


# In[ ]:


def mask2rle(mask):
    if np.sum(mask) == 0: return ''
    ar = mask.flatten(order='F')
    EncodedPixel = ''
    l = 0
    for i in range(len(ar)):
        if ar[i] == 0:
            if l > 0:
                if EncodedPixel != '': EncodedPixel += ' '
                EncodedPixel += str(st+1)+' '+str(l)
                l = 0
        else: # == 1
            if l == 0: st = i
            l += 1
    return EncodedPixel


# In[1]:


import cv2
def remove_small_one(predict, min_size):
    H,W = predict.shape
    num_component, component = cv2.connectedComponents(predict.astype(np.uint8))
    predict = np.zeros((H,W), np.bool)
    for c in range(1,num_component):
        p = (component==c)
        if p.sum()>min_size:
            predict[p] = True
    return predict


# In[2]:


def sharpen(p,t=0.5):
        if t!=0:
            return p**t
        else:
            return p


# In[ ]:


import random
import time
# test_stage
trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)
                )
            ]
        )

test_dir = "../input/severstal-steel-defect-detection/test_images/"

imglists = glob.glob(test_dir + '/*g')
oriims = []
preds = []
random.shuffle(imglists)
ImageId_ClassIds = []
EncodedPixels = []

augs = ['flip_lr', 'flip_ud']
from tqdm import tqdm
thresholds = [0.4, 0.4, 0.4, 0.4]
min_area = [600, 600, 1000, 2000]
# min_area = [1, 1, 1, 1]
t1 = time.time()
for i, item in enumerate(tqdm(imglists)):
    timg = cv2.imread(item)[:, :, ::-1]
    img = mx.nd.array([timg])
    input_img = trans(img)
    num_aug = 0
    
    if 1:
        out = unet(input_img.as_in_context(ctx))
        
        out = F.softmax(out, axis=1)
        # out = F.where(out > 0.5, out, F.zeros_like(out))
        pred_inds = F.argmax(out, axis=1)
        oriims.append(timg)
        preds.append(pred_inds)
        out_mask = sharpen(out, 0)
        
        num_aug += 1
    if 'flip_lr' in augs:
        input_img_lr = F.flip(input_img, axis=3)
        out = unet(input_img_lr.as_in_context(ctx))
        out = F.softmax(out, axis=1)
        # out = F.where(out > 0.5, out, F.zeros_like(out))
        out_mask += sharpen(F.flip(out, axis=3))
        num_aug += 1

    # if 'flip_ud' in augs:
    #     input_img_lr = F.flip(input_img, axis=2)
    #     out = unet(input_img_lr.as_in_context(ctx))
    #     out = F.softmax(out, axis=1)
    #     # out = F.where(out > 0.5, out, F.zeros_like(out))
    #     out_mask += sharpen(F.flip(out, axis=2))
    #     num_aug += 1

    
    out_mask = out_mask * 1.0 / num_aug
    out = out_mask[:, 1:, :, :].asnumpy()
    ImageId = item.split('/')[-1]
    pred_inds = pred_inds.asnumpy()
    for j in range(4):
        Id = ImageId + '_'+str(j+1)
        tmp_mask = np.where(out[:, j, :, :] > thresholds[j], 1.0, 0)
        # tmp_mask = remove_small_one(tmp_mask[0], min_size=100).astype(np.float)
#         tmp_mask = np.where(pred_inds[0, :, :]==(j+1), 1, 0)
        if np.sum(tmp_mask) < min_area[j]:
            tmp_mask = np.zeros_like(tmp_mask)

        if np.sum(tmp_mask) < 10:
            EncodedPixel = ''
        else:
            EncodedPixel = mask2rle(tmp_mask)

        ImageId_ClassIds.append(Id)
        EncodedPixels.append(EncodedPixel)
dur = time.time() - t1
print("cost time:{}".format(dur))


# In[ ]:


submission =  pd.read_csv("../input/severstal-steel-defect-detection/sample_submission.csv")
print(len(ImageId),len(submission['ImageId_ClassId']))
# len(set(submission['ImageId_ClassId'])-set(Ids))
# assert set(Ids) == set(submission['ImageId_ClassId'])

for i, encoded in zip(ImageId_ClassIds,EncodedPixels):
    submission.loc[submission['ImageId_ClassId']==i,["EncodedPixels"]] =  encoded

submission.to_csv('submission.csv',index=False)


# In[ ]:


submission.head(10)


# In[ ]:


fig, ax1 = plt.subplots(figsize=(50, 50))
for i, (timg, pred_inds) in enumerate(zip(oriims[:100], preds[:100])):
#     plt.subplot(len(oriims[:100])*2, 1, i*2+1)
#     plt.imshow(timg)
#     plt.subplot(len(oriims[:100])*2, 1, i*2+2)
#     plt.imshow(pred_inds[0].asnumpy())
    seg_map = np.expand_dims(pred_inds[0].asnumpy(), axis=2)
    seg_map_3c=np.repeat(seg_map, 3, 2)*255
    h, w = timg.shape[:2]
    seg_map_3c = cv2.resize(seg_map_3c, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    att_im = cv2.addWeighted(seg_map_3c.astype(np.uint8), 0.5, timg, 0.5, 0.0)
    if i > 10:
        break
    plt.subplot(11, 1, i+1)
    plt.imshow(att_im)

