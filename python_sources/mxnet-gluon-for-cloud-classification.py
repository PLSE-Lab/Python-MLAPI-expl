#!/usr/bin/env python
# coding: utf-8

# **in this kernel i will try to implement [this kernel](https://www.kaggle.com/jiageng/segmentation-cls) for understanding cloud competiton and for inference [this kernel](https://www.kaggle.com/jiageng/segmentation-cls) can be used**
# 
# # please upvote that original authors kernel by visiting those 2 links above!

# In[ ]:


import os
print(os.listdir('../input/understanding_cloud_organization'))


# In[ ]:


import numpy as np, pandas as pd, os, gc
import matplotlib.pyplot as plt, time
from PIL import Image 
import warnings
import random
warnings.filterwarnings("ignore")

path = '../input/understanding_cloud_organization/'
train = pd.read_csv(path + 'train.csv')

# RESTRUCTURE TRAIN DATAFRAME
train['ImageId'] = train['Image_Label'].map(lambda x: x.split('.')[0]+'.jpg')
train2 = pd.DataFrame({'ImageId':train['ImageId'][::4]})
train2['e1'] = train['EncodedPixels'][::4].values
train2['e2'] = train['EncodedPixels'][1::4].values
train2['e3'] = train['EncodedPixels'][2::4].values
train2['e4'] = train['EncodedPixels'][3::4].values
train2.reset_index(inplace=True,drop=True)
train2.fillna('',inplace=True); 
train2['count'] = np.sum(train2.iloc[:,1:]!='',axis=1).values

indexes = list(range(len(train2)))
random.shuffle(indexes)
train_ratio = 0.99
partio = int(len(train2) * train_ratio)
train_indexes = indexes[:partio]
val_indexes = indexes[partio:]
train_df = train2.iloc[train_indexes, :]
val_df = train2.iloc[val_indexes, :]


# ### Data augument with albumentations

# In[ ]:


from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, PadIfNeeded, RandomCrop,
    RGBShift, RandomBrightness, RandomContrast, VerticalFlip
)
train_augmentator = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.03, scale_limit=0,
                         rotate_limit=(-3, 3), border_mode=0, p=0.75),
        RandomBrightness(limit=(-0.25, 0.25), p=0.75),
        RandomContrast(limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)


# In[ ]:


h = 400
w = 400


# In[ ]:


import mxnet as mx
from mxnet.gluon import data, HybridBlock, nn
import pandas as pd
import cv2
import os
import numpy as np
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo import vision
from mxnet.lr_scheduler import CosineScheduler
from mxnet.gluon import loss, Trainer
from mxnet import autograd
import random
from PIL import Image, ImageOps, ImageFilter
from mxnet import nd as F, lr_scheduler as lrs
from mxnet.gluon.contrib.estimator import Estimator
import gluoncv.model_zoo  as gm

def scale_func(image_shape):
    return random.uniform(0.5, 1.2)


class cloudDataset(data.Dataset):
    def __init__(self, df, img_dir, debug=False):
        
        self.train_df = df
        self.root_dir = img_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)
                )
            ]
        )
        
        self.debug = debug
        
    def __getitem__(self, i):
        if self.debug:
            curr_df = self.train_df.head(20)
        masks = np.zeros((h, w), np.uint8)
        img_names = []
        item = self.train_df.iloc[i, :]
        img_name = item['ImageId']
        defect_label = np.zeros((1, 4), dtype=np.float32)
        for j in range(4):
            curr_item = item["e{}".format(j+1)]
            if len(curr_item) > 0:

                rle_pixels = curr_item
                label = rle_pixels.split(" ")
                positions = list(map(int, label[0::2]))
                length = list(map(int, label[1::2]))
                mask = np.zeros(h * w, dtype=np.uint8)
                for pos, le in zip(positions, length):
                    mask[pos - 1:(pos + le - 1)] = j+1
                count = np.sum(np.where(mask==(j+1), 1, 0))
                if count < 8:
                    mask = np.where(mask==(j+1), -1, 0)
                defect_label[:, j] = 1
                masks[ :, :] = masks[ :, :] + mask.reshape(h, w, order='F')
                
        oimg = cv2.imread(os.path.join(self.root_dir, img_name))[:, :, ::-1]
        # oimg, masks = self.rescale_sample(oimg, masks)
        aug_out = train_augmentator(image=oimg, mask=masks)
        oimg = aug_out['image']
        masks = aug_out['mask']
      
        img = F.array(oimg)
        img = self.transform(img)
        
        if self.debug:
            return img, F.array(masks[::4, ::4]), oimg, masks, curr_df
        else:
            return img, F.array(masks), F.array(defect_label)
        
    def __len__(self):
        return len(self.train_df)


    def rescale_sample(self, image, mask):

        scale = scale_func(image.shape)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        new_size = (image.shape[1], image.shape[0])

        mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

        return image, mask


# In[ ]:


# for test
import matplotlib.pyplot as plt
csv_file = '../input/understanding_cloud_organization/train.csv'
img_dir = '../input/understanding_cloud_organization/train_images/'
cloud_dataset = cloudDataset(train2, img_dir, debug=True)
print(len(cloud_dataset))
_, mm, im, mask, curr_df = cloud_dataset[16]
plt.figure(figsize=(20, 20))
plt.subplot(2, 1, 1)
plt.imshow(im)
plt.subplot(2, 1, 2)
plt.imshow(mask[::4, ::4])
mm.flatten().shape


# ### FPN-based segmentation

# In[ ]:


from gluoncv.model_zoo.resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s
import mxnet as mx

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

class ResNetSteel(mx.gluon.HybridBlock):
    def __init__(self, backbone= 'resnet50', num_classes=4, backbone_lr_mult=0.1, **kwargs):
        super(ResNetSteel, self).__init__()

        self.backbone_name = backbone
        self.backbone_lr_mult = backbone_lr_mult
        self._kwargs = kwargs

        with self.name_scope():
            self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False, dilated=False, **kwargs)

            self.head = Classification_head(output_channels=256, num_classes=num_classes)

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
        logits = self.head(c4)

        return logits

class Classification_head(HybridBlock):
    def __init__(self, output_channels=256, num_classes=4):
        super(Classification_head, self).__init__()

        with self.name_scope():
            self.cls_head = nn.HybridSequential()
            self.cls_head.add(ConvBlock(output_channels, kernel_size=1))
            self.cls_head.add(nn.GlobalAvgPool2D())
            self.cls_head.add(nn.Conv2D(num_classes, kernel_size=1))

    def hybrid_forward(self, F, x):
        logits = self.cls_head(x)

        return F.squeeze(logits)


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


unet = ResNetSteel(num_classes=4)
unet.collect_params().initialize()
unet.load_pretrained_weights()
a = mx.nd.normal(shape=(4, 3, h, w))
logits = unet(a)
print(logits.shape)


# ### loss function
# - focal loss
# - normalize focal loss
# - dice loss
# - bce loss

# In[ ]:


import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like

class NormalizedFocalLossSoftmax(Loss):
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, detach_delimeter=True, gamma=2, eps=1e-10, **kwargs):
        super(NormalizedFocalLossSoftmax, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._eps = eps
        self._gamma = gamma
        self._k_sum = 0

    def hybrid_forward(self, F, pred, label):
        label = F.expand_dims(label, axis=1)
        softmaxout = F.softmax(pred, axis=1)

        t = label != self._ignore_label
        pt = F.pick(softmaxout, label, axis=1, keepdims=True)
        pt = F.where(t, pt, F.ones_like(pt))
        beta = (1 - pt) ** self._gamma

        t_sum = F.cast(F.sum(t, axis=(-2, -1), keepdims=True), 'float32')
        beta_sum = F.sum(beta, axis=(-2, -1), keepdims=True)
        mult = t_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = F.broadcast_mul(beta, mult)
        self._k_sum = 0.9 * self._k_sum + 0.1 * mult.asnumpy().mean()

        loss = -beta * F.log(F.minimum(pt + self._eps, 1))

        if self._size_average:
            bsum = F.sum(t_sum, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (bsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)


class NormalizedFocalLossSigmoid(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, size_average=True, detach_delimeter=True,
                 eps=1e-12, scale=1.0,
                 ignore_label=-1, **kwargs):
        super(NormalizedFocalLossSigmoid, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label

        self._scale = scale
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._k_sum = 0

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        one_hot = label > 0
        t = F.ones_like(one_hot)

        if not self._from_logits:
            pred = F.sigmoid(pred)

        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        pt = F.where(one_hot, pred, 1 - pred)
        pt = F.where(label != self._ignore_label, pt, F.ones_like(pt))

        beta = (1 - pt) ** self._gamma

        t_sum = F.sum(t, axis=(-2, -1), keepdims=True)
        beta_sum = F.sum(beta, axis=(-2, -1), keepdims=True)
        mult = t_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = F.broadcast_mul(beta, mult)

        ignore_area = F.sum(label == -1, axis=0, exclude=True).asnumpy()
        sample_mult = F.mean(mult, axis=0, exclude=True).asnumpy()
        if np.any(ignore_area == 0):
            self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

        loss = -alpha * beta * F.log(F.minimum(pt + self._eps, 1))
        sample_weight = label != self._ignore_label

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            bsum = F.sum(sample_weight, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (bsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return self._scale * loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)


class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.sigmoid(pred)

        one_hot = label > 0
        pt = F.where(one_hot, pred, 1 - pred)

        t = label != -1
        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * F.log(F.minimum(pt + self._eps, 1))
        sample_weight = label != -1

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            tsum = F.sum(label == 1, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (tsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return self._scale * loss


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, grad_scale=1.0, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average
        self._grad_scale = grad_scale

    def hybrid_forward(self, F, pred, label):
        softmaxout = F.SoftmaxOutput(
            pred, label.astype(pred.dtype), ignore_label=self._ignore_label,
            multi_output=self._sparse_label,
            use_ignore=True, normalization='valid' if self._size_average else 'null',
            grad_scale=self._grad_scale,
        )
        loss = -F.pick(F.log(softmaxout), label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class SigmoidBinaryCrossEntropyLoss(Loss):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def hybrid_forward(self, F, pred, label):
        label = _reshape_like(F, label, pred)
        sample_weight = label != self._ignore_label
        label = F.where(sample_weight, label, F.zeros_like(label))

        if not self._from_sigmoid:
            loss = F.relu(pred) - pred * label +                 F.Activation(-F.abs(pred), act_type='softrelu')
        else:
            eps = 1e-12
            loss = -(F.log(pred + eps) * label
                     + F.log(1. - pred + eps) * (1. - label))

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


# In[ ]:


def compute_iou(label, pred):
    union = np.logical_or(label, pred)
    intersection = np.logical_and(label, pred)
    iou = intersection / (union + 1e-5)
    return np.mean(iou)

def iou_metric(labels, preds):
    
#     labels = F.array(labels)
#     preds = F.array(preds)
    labels = labels.asnumpy()
    preds = F.argmax(F.softmax(preds, axis=1), axis=1).asnumpy()
    ious = []
    for i in range(5):
        curr_pred = np.where(preds==i, 1, 0)
        curr_labels = np.where(labels==i, 1, 0)
        curr_iou = compute_iou(curr_labels, curr_pred)
        ious.append(curr_iou)
    mean_iou = np.mean(ious)
    ious.append(mean_iou)
#     print("IOU_INFO:: bg:{}, 1:{}, 2:{}, 3:{}, 4:{}, mean_iou:{}".format(*ious))
    cls = ['bg', '1', '2', '3', '4', 'mean_iou']
    return {k:v for k, v in zip(cls, ious)}


# In[ ]:


def training(epoch, data, net, cls_loss, trainer, ctx):
    train_loss = 0.0
    hybridize = False
    tbar = tqdm(data)
    for i, batch_data in enumerate(tbar):
        image, mask, label = batch_data
        image = image.as_in_context(ctx)
        mask = mask.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            logits = net(image)
            cls_losses = cls_loss(logits, label)
            losses = cls_losses
        losses.backward()
        global_step = epoch * len(data) + i
        trainer.step(len(batch_data))

        batch_loss = sum(loss.asnumpy().mean() for loss in losses) / len(losses)
        train_loss += batch_loss
       
        if i % 6:
            tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.6f}')


# In[ ]:


def evaluation(data, net, ctx):
    
    val_acc = 0.0
    hybridize = False
    tbar = tqdm(data)
    for i, batch_data in enumerate(tbar):
        image, mask, label = batch_data
        image = image.as_in_context(ctx)
        mask = mask.as_in_context(ctx)
        label = label.as_in_context(ctx)
        logits = net(image)
        probs = F.sigmoid(logits)
        probs = F.where(probs > 0.5, F.ones_like(probs), F.zeros_like(probs))
        val_acc += F.mean(label==probs).asscalar()
        if i % 6:
            tbar.set_description(f'val_accs {val_acc/(i+1):.6f}')
    return val_acc * 1.0 /(i+1)


# In[ ]:


import os
from tqdm import tqdm
def train_from_manual(train_df, val_df, img_dir, batch_size, epoches, lr=0.001, ctx=mx.cpu()):
    # TODO: finish trainer .etc, add ctx
    cloud_dataset = cloudDataset(train_df, img_dir)
    cloud_data = data.DataLoader(cloud_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    
    val_steel_dataset = cloudDataset(val_df, img_dir)
    val_steel_data = data.DataLoader(val_steel_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    
    normal_focal_loss = NormalizedFocalLossSoftmax(ignore_label=-1, gamma=1)
    bce_loss = SigmoidBinaryCrossEntropyLoss()
    
    unet = ResNetSteel(num_classes=4)
    unet.initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=2), ctx=ctx)
    unet.load_pretrained_weights()
    for k, v in unet.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    lr_sche = lrs.FactorScheduler(step=5, base_lr=lr, factor=0.7,  warmup_steps=2, warmup_begin_lr=0.00002)
    trainer = Trainer(unet.collect_params(), 'adam', 
                        {'learning_rate': lr,
                         'wd':1e-5,
#                          'lr_scheduler': lr_sche
                        })
    for epoch in range(epoches):
        max_acc = -1
        if epoch in [10, 15, 20, 25, 35]:
            lr = lr * 0.7
            trainer.set_learning_rate(lr=lr)
        training(epoch, cloud_data, unet, bce_loss, trainer, ctx)
        if epoch % 2 == 0:
            val_acc = evaluation(val_steel_data, unet, ctx)
            if val_acc > max_acc:
                print("acc from {} improve to {}".format(max_acc, val_acc))
                max_acc = val_acc
                
            unet.save_parameters('unet_{}_{}.params'.format(epoch, max_acc))


# In[ ]:


batch_size = 4
csv_file = '../input/understanding_cloud_organization/train.csv'
img_dir = '../input/understanding_cloud_organization/train_images/'

epoches = 1
train_from_manual(train_df, val_df, img_dir, batch_size, epoches, ctx=mx.gpu())


# In[ ]:


import pandas as pd
sample_submission = pd.read_csv("../input/understanding_cloud_organization/sample_submission.csv")
train = pd.read_csv("../input/understanding_cloud_organization/train.csv")


# In[ ]:





# In[ ]:




