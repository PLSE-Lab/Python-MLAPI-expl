#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import cv2
import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torchvision


# In[ ]:


# Hyper-params
MODEL_PATH = '/kaggle/input/fpnup50/model_best.pth'
input_size = 512
IN_SCALE = 1024 // input_size
MODEL_SCALE = 4


# In[ ]:


DIR_INPUT = '/kaggle/input/global-wheat-detection'
DIR_TEST = f'{DIR_INPUT}/test'


# In[ ]:


def pred2box(hm, regr, thresh=0.99):
    # make binding box from heatmaps
    # thresh: threahold for logits.
    
    # get center
    pred = hm > thresh
    pred_center = np.where(hm > thresh)
    # get regressions
    pred_r = regr[:, pred].T
    
    # wrap as boxes
    # [xmin, ymin, width, height]
    # size as original image
    boxes = []
    scores = hm[pred]
    for i, b in enumerate(pred_r):
        arr = np.array([pred_center[1][i] * MODEL_SCALE - b[0] * input_size // 2, 
                        pred_center[0][i] * MODEL_SCALE - b[1] * input_size // 2,
                        int(b[0] * input_size), int(b[1] * input_size)])
        arr = np.clip(arr, 0, input_size)
        # filter
        # if arr[0] < 0 or arr[1] < 0 or arr[0] > input_size or arr[1] > input_size: pass
        boxes.append(arr)
    return np.asarray(boxes), scores


# In[ ]:


# functions for plotting results
# def showbox(img, hm, regr, thresh=0.9):
#     boxes, _ = pred2box(hm, regr, thresh=thresh)
#     print('preds:', boxes.shape)
#     sample = img
    
#     for box in boxes:
#         # upper-left, lower-right
#         cv2.rectangle(sample,
#                      (int(box[0]), int(box[1]+box[3])),
#                      (int(box[0]+box[2]), int(box[1])),
#                      (220, 0, 0), 3)
#     return sample
def boxshow(img, boxes, color=(220, 0, 0)):
    sample = img
    for box in boxes:
        cv2.rectangle(sample,
                      (int(box[0]), int(box[1] + box[3])),
                      (int(box[0] + box[2]), int(box[1])),
                      color, 3)
    return sample


# In[ ]:


# pool duplicates
def pool(data):
    stride = 3
    for y in np.arange(1, data.shape[1]-1, stride):
        for x in np.arange(1, data.shape[0]-1, stride):
            a_2d = data[x-1:x+2, y-1:y+2]
            max = np.asarray(np.unravel_index(np.argmax(a_2d), a_2d.shape))
            for c1 in range(3):
                for c2 in range(3):
                    if not (c1 == max[0] and c2 == max[1]):
                        data[x+c1-1, y+c2-1] = -1
    return data

# NMS is required to remove duplicate boxes
def nms(boxes, scores, overlap=0.25, top_k=200):
#     print(boxes.shape)
#     print(scores.shape)
    scores = torch.from_numpy(scores)
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    boxes = boxes.reshape(-1, 4)
    boxes = torch.from_numpy(np.array([boxes[:,0], boxes[:,1], boxes[:,0]+boxes[:,2], boxes[:,1]+boxes[:,3]]).T.reshape(-1, 4))
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2-x1, y2-y1)
    
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()
    
    v, idx = scores.sort(0)
    
    idx = idx[-top_k:]
    
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        
        if idx.size(0) == 1:
            break
            
        idx = idx[:-1]
        
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)
        
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])
        
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)
        
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1
        
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)
        
        inter = tmp_w * tmp_h
        
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union
        
        idx  = idx[IoU.le(overlap)]
        
    return keep.numpy(), count


# In[ ]:


from torchvision import transforms

# Submission
class WheatDatasetTest(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.img_id = os.listdir(self.image_dir)
        
    def __len__(self):
        return len(self.img_id)
    
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_dir, self.img_id[idx]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_size, input_size))
        img = img.astype(np.float32) / 255.0
        img = img.transpose([2, 0, 1])
        return img, self.img_id[idx]


# In[ ]:


testdataset = WheatDatasetTest(DIR_TEST)
test_loader = torch.utils.data.DataLoader(testdataset, batch_size=16, shuffle=False, num_workers=0)


# In[ ]:


import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = norm_layer(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = norm_layer(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        # self.shrink = nn.Sequential(
        #     nn.Conv2d(in_ch, in_ch//2, 3, padding=1),
        #     nn.BatchNorm2d(in_ch // 2),
        #     nn.ReLU(inplace=True)
        # )
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # x1 = self.shrink(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


class centernet(nn.Module):
    def __init__(self, num_class=1):
        super(centernet, self).__init__()
        basemodel = torchvision.models.resnet18(pretrained=True)
        basemodel = nn.Sequential(*list(basemodel.children())[:-2])
        self.basemodel = basemodel

        num_ch = 512
        self.up1 = up(num_ch, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 256)

        # self.outc = nn.Conv2d(256, num_class, 1)
        # self.outr = nn.Conv2d(256, 2, 1)
        self.outc = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, num_class, kernel_size=1, stride=1, padding=0))
        self.outr = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.basemodel(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        hm = self.outc(x)
        wh = self.outr(x)
        ret = {'hm': hm, 'wh': wh}
        return ret


class CenterNetRes34(nn.Module):
    def __init__(self, num_class=1):
        super(CenterNetRes34, self).__init__()
        basemodel = torchvision.models.resnet34(pretrained=True)
        basemodel = nn.Sequential(*list(basemodel.children())[:-2])
        self.basemodel = basemodel

        num_ch = 512
        self.up1 = up(num_ch, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 256)

        self.outc = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Conv2d(128, num_class, kernel_size=1, stride=1, padding=0))
        self.outr = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.basemodel(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        hm = self.outc(x)
        wh = self.outr(x)
        ret = {'hm': hm, 'wh': wh}
        return ret


## FPN
class Botttleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, act = nn.LeakyReLU(negative_slope=0.1, inplace=True)):
        super(Botttleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.act = act

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class UpSampleConcat(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2):
        super(UpSampleConcat, self).__init__()
        self.conv = double_conv(in_ch*2, out_ch)
        self.up = Upsampler(scale, in_ch)

    def forward(self, x, y):
        _, _, H, W = y.size()
        # x1 = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x1 = self.up(x)
        out = self.conv(torch.cat([x1, y], dim=1))
        return out


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=True, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
                elif act == 'leakylu':
                    m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
            elif act == 'leakylu':
                m.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class FPN(nn.Module):
    def __init__(self, block, num_blocks, num_class=1, act=nn.LeakyReLU(negative_slope=0.1, inplace=True)):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.act = act

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # upsample Concat
        self.up1 = UpSampleConcat(256, 256)
        self.up2 = UpSampleConcat(256, 256)
        self.up3 = UpSampleConcat(256, 256)

        self.outc = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, num_class, kernel_size=1, stride=1, padding=0))
        self.outr = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def _upsample_add(self, x, y):
    #     _, _, H, W = y.size()
    #     return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        c1 = self.act(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.toplayer(c5)
        # p4 = self._upsample_add(p5, self.latlayer1(c4))
        # p3 = self._upsample_add(p4, self.latlayer2(c3))
        # p2 = self._upsample_add(p3, self.latlayer3(c2))
        p4 = self.up1(p5, self.latlayer1(c4))
        p3 = self.up2(p4, self.latlayer2(c3))
        p2 = self.up3(p3, self.latlayer3(c2))

        hm = self.outc(p2)
        wh = self.outr(p2)
        ret = {'hm': hm, 'wh': wh}
        return ret


def FPNMaker():
    return FPN(Botttleneck, [2, 2, 2, 2], num_class=1)


# In[ ]:


# model = centernet(num_class=1)
model = FPNMaker()


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)


# In[ ]:


# model.load_state_dict(torch.load(MODEL_PATH))
# model.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc:storage))
model_old = torch.load(MODEL_PATH, map_location=lambda storage, loc:storage)


# In[ ]:


state_dict_ = model_old['state_dict']


# In[ ]:


model.load_state_dict(state_dict_, strict=False)


# In[ ]:


def format_prediction_string(boxes, scores):
    pred_strings = []
    for s, b in zip(scores, boxes.astype(int)):
        # xmin, ymin, w, h
        pred_strings.append(f'{s:.4f} {b[0]*IN_SCALE} {b[1]*IN_SCALE} {b[2]*IN_SCALE} {b[3]*IN_SCALE}')
    return ' '.join(pred_strings)


# In[ ]:


model.eval()


# In[ ]:


thresh = 0.45
results = []
for images, image_ids in tqdm(test_loader):
    images = images.to(device)
#     print(images.shape)
    with torch.no_grad():
        outputs = model(images)
#     print(outputs)
        
    for hm, regr, image_id in zip(outputs['hm'], outputs['wh'], image_ids):
#         print(hm.shape)
    
        # process predictions
        hm = hm.cpu().numpy().squeeze(0)
        regr = regr.cpu().numpy()
        hm = torch.sigmoid(torch.from_numpy(hm)).numpy()
        hm = pool(hm)
        
        boxes, scores = pred2box(hm, regr, thresh)
#         print(boxes.shape)
#         print(scores.shape)
        # Filter by nms
        keep, count = nms(boxes, scores, overlap=0.15)
        boxes = boxes[keep[:count]]
        scores = scores[keep[:count]]
        
        preds_sorted_idx = np.argsort(scores)[::-1]
        boxes_sorted = boxes[preds_sorted_idx]
        scores_sorted = scores[preds_sorted_idx]
        
#         img = cv2.imread(os.path.join(DIR_INPUT, 'test', image_id))
#         img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (input_size, input_size))
#         sample = boxshow(img, boxes_sorted, color=(220, 0, 0))
#         plt.imshow(sample)
#         plt.show()
        
        
        result = {
            'image_id': image_id[:-4],
            'PredictionString': format_prediction_string(boxes, scores)
        }
        results.append(result)


# In[ ]:


test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.head(10)


# In[ ]:


test_df.to_csv('submission.csv', index=False)


# In[ ]:




