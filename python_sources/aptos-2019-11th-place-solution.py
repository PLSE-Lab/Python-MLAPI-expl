#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import random
import math
import subprocess
from glob import glob
from collections import OrderedDict
import numpy as np
import cv2
from skimage import measure
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
from sklearn.linear_model import LinearRegression

import torch
from torch import nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torch.optim.optimizer import Optimizer, required

sys.path.append("../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/")
import pretrainedmodels


# In[ ]:


def scale_radius(src, img_size, padding=False):
    x = src[src.shape[0] // 2, ...].sum(axis=1)
    r = (x > x.mean() / 10).sum() // 2
    yx = src.sum(axis=2)
    region_props = measure.regionprops((yx > yx.mean() / 10).astype('uint8'))
    yc, xc = np.round(region_props[0].centroid).astype('int')
    x1 = max(xc - r, 0)
    x2 = min(xc + r, src.shape[1] - 1)
    y1 = max(yc - r, 0)
    y2 = min(yc + r, src.shape[0] - 1)
    dst = src[y1:y2, x1:x2]
    dst = cv2.resize(dst, dsize=None, fx=img_size/(2*r), fy=img_size/(2*r))
    if padding:
        pad_x = (img_size - dst.shape[1]) // 2
        pad_y = (img_size - dst.shape[0]) // 2
        dst = np.pad(dst, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'constant')
    return dst

    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None, img_size=288, save_img=True):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
        self.save_img = save_img

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]
        
        if 'train' in img_path:
            img = cv2.imread('../input/aptos2019-dataset/images_288_scaled/images_288_scaled/%s' %os.path.basename(img_path))
        
        else:
            if os.path.exists('processed/%s' %os.path.basename(img_path)):
                img = cv2.imread('processed/%s' %os.path.basename(img_path))

            else:
                img = cv2.imread(img_path)
                try:
                    img = scale_radius(img, img_size=self.img_size, padding=False)
                except Exception as e:
                    img = img
                if self.save_img:
                    cv2.imwrite('processed/%s' %os.path.basename(img_path), img)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


    def __len__(self):
        return len(self.img_paths)


def get_model(model_name='resnet18', num_outputs=None, pretrained=True,
              freeze_bn=False, dropout_p=0, **kwargs):

    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__[model_name](num_classes=1000,
                                                  pretrained=pretrained)

    if 'dpn' in model_name:
        in_channels = model.last_linear.in_channels
        model.last_linear = nn.Conv2d(in_channels, num_outputs,
                                      kernel_size=1, bias=True)
    else:
        if 'resnet' in model_name:
            model.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            model.avg_pool = nn.AdaptiveAvgPool2d(1)
        in_features = model.last_linear.in_features
        if dropout_p == 0:
            model.last_linear = nn.Linear(in_features, num_outputs)
        else:
            model.last_linear = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_outputs),
            )

    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    return model


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def quadratic_weighted_kappa(y_pred, y_true):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    if y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return metrics.cohen_kappa_score(y_pred, y_true, weights='quadratic')


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output.view(-1), target.float())

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        thrs = [0.5, 1.5, 2.5, 3.5]
        output[output < thrs[0]] = 0
        output[(output >= thrs[0]) & (output < thrs[1])] = 1
        output[(output >= thrs[1]) & (output < thrs[2])] = 2
        output[(output >= thrs[2]) & (output < thrs[3])] = 3
        output[output >= thrs[3]] = 4
        
        target[target < thrs[0]] = 0
        target[(target >= thrs[0]) & (target < thrs[1])] = 1
        target[(target >= thrs[1]) & (target < thrs[2])] = 2
        target[(target >= thrs[2]) & (target < thrs[3])] = 3
        target[target >= thrs[3]] = 4
        
        score = quadratic_weighted_kappa(output, target)

        losses.update(loss.item(), input.size(0))
        scores.update(score, input.size(0))

    return losses.avg, scores.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output.view(-1), target.float())
        
            thrs = [0.5, 1.5, 2.5, 3.5]
            output[output < thrs[0]] = 0
            output[(output >= thrs[0]) & (output < thrs[1])] = 1
            output[(output >= thrs[1]) & (output < thrs[2])] = 2
            output[(output >= thrs[2]) & (output < thrs[3])] = 3
            output[output >= thrs[3]] = 4
            score = quadratic_weighted_kappa(output, target)

            losses.update(loss.item(), input.size(0))
            scores.update(score, input.size(0))

    return losses.avg, scores.avg


# In[ ]:


os.makedirs('processed', exist_ok=True)


# # Pseudo Labeling

# In[ ]:


pseudo_probs = {}

aptos2019_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
aptos2019_img_paths = '../input/aptos2019-blindness-detection/train_images/' + aptos2019_df['id_code'].values + '.png'
aptos2019_labels = aptos2019_df['diagnosis'].values

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
dir_name = '../input/aptos2019-blindness-detection/test_images'
test_img_paths = dir_name + '/' + test_df['id_code'].values + '.png'
test_labels = np.zeros(len(test_img_paths))

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_set = Dataset(
    test_img_paths,
    test_labels,
    transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=32,
    shuffle=False,
    num_workers=2)


# ## SE-ResNeXt50_32x4d

# In[ ]:


# create model
model = get_model(model_name='se_resnext50_32x4d',
                  num_outputs=1,
                  pretrained=False,
                  freeze_bn=True,
                  dropout_p=0)
model = model.cuda()
model.eval()

probs = []
for fold in range(5):
    print('Fold [%d/%d]' %(fold+1, 5))

    model.load_state_dict(torch.load('../input/se-resnext50-32x4d-080922/model_%d.pth' % (fold+1)))

    probs_fold = []
    with torch.no_grad():
        for i, (input, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
            input = input.cuda()
            output = model(input)

            probs_fold.extend(output.data.cpu().numpy()[:, 0])
    probs_fold = np.array(probs_fold)
    probs.append(probs_fold)

probs = np.mean(probs, axis=0)
pseudo_probs['se_resnext50_32x4d'] = probs

del model
torch.cuda.empty_cache()
get_ipython().system('nvidia-smi')


# ## SE-ResNeXt101_32x4d

# In[ ]:


# create model
model = get_model(model_name='se_resnext101_32x4d',
                  num_outputs=1,
                  pretrained=False,
                  freeze_bn=True,
                  dropout_p=0,
                  )
model = model.cuda()
model.eval()

probs = []
for fold in range(5):
    print('Fold [%d/%d]' %(fold+1, 5))

    model.load_state_dict(torch.load('../input/se-resnext101-32x4d-081208/model_%d.pth' % (fold+1)))

    probs_fold = []
    with torch.no_grad():
        for i, (input, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
            input = input.cuda()
            output = model(input)

            probs_fold.extend(output.data.cpu().numpy()[:, 0])
    probs_fold = np.array(probs_fold)
    probs.append(probs_fold)

probs = np.mean(probs, axis=0)
pseudo_probs['se_resnext101_32x4d'] = probs

del model
torch.cuda.empty_cache()
get_ipython().system('nvidia-smi')


# ## SENet154

# In[ ]:


# create model
model = get_model(model_name='senet154',
                  num_outputs=1,
                  pretrained=False,
                  freeze_bn=True,
                  dropout_p=0)
model = model.cuda()
model.eval()

probs = []
for fold in range(5):
    print('Fold [%d/%d]' %(fold+1, 5))

    model.load_state_dict(torch.load('../input/senet154-082510/model_%d.pth' % (fold+1)))

    probs_fold = []
    with torch.no_grad():
        for i, (input, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
            input = input.cuda()
            output = model(input)

            probs_fold.extend(output.data.cpu().numpy()[:, 0])
    probs_fold = np.array(probs_fold)
    probs.append(probs_fold)

probs = np.mean(probs, axis=0)
pseudo_probs['senet154'] = probs

del model
torch.cuda.empty_cache()
get_ipython().system('nvidia-smi')


# # Train & Inference

# In[ ]:


l1_probs = {}

train_transform = []
train_transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.RandomAffine(
        degrees=(-180, 180),
        scale=(0.8889, 1.0),
        shear=(-36, 36),
    ),
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0,
        contrast=(0.9, 1.1),
        saturation=0,
        hue=0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ## SE-ResNeXt50_32x4d

# ### Train

# In[ ]:


aptos2019_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
aptos2019_img_paths = '../input/aptos2019-blindness-detection/train_images/' + aptos2019_df['id_code'].values + '.png'
aptos2019_labels = aptos2019_df['diagnosis'].values

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
test_img_paths = '../input/aptos2019-blindness-detection/test_images/' + test_df['id_code'].values + '.png'
test_labels = 0.4 * pseudo_probs['se_resnext50_32x4d'] + 0.3 * pseudo_probs['se_resnext101_32x4d'] + 0.3 * pseudo_probs['senet154']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
kf = KFold(n_splits=5, shuffle=True, random_state=41)
img_paths = []
labels = []
for (train_idx1, val_idx1), (train_idx2, val_idx2) in zip(skf.split(aptos2019_img_paths, aptos2019_labels), kf.split(test_img_paths)):
    img_paths.append((np.hstack((aptos2019_img_paths[train_idx1], test_img_paths[val_idx2])), aptos2019_img_paths[val_idx1]))
    labels.append((np.hstack((aptos2019_labels[train_idx1], test_labels[val_idx2])), aptos2019_labels[val_idx1]))


# In[ ]:


# create model
model = get_model(model_name='se_resnext50_32x4d',
                  num_outputs=1,
                  pretrained=False,
                  freeze_bn=True,
                  dropout_p=0,
                  )
model = model.cuda()

criterion = nn.MSELoss().cuda()

best_losses = []
best_scores = []

for fold, ((train_img_paths, val_img_paths), (train_labels, val_labels)) in enumerate(zip(img_paths, labels)):
    print('Fold [%d/%d]' %(fold+1, len(img_paths)))

    # train
    train_set = Dataset(
        train_img_paths,
        train_labels,
        transform=train_transform,
        img_size=288,
        save_img=True)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=2)

    val_set = Dataset(
        val_img_paths,
        val_labels,
        transform=val_transform,
        save_img=True)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=2)

    model.load_state_dict(torch.load('../input/se-resnext50-32x4d-080922/model_%d.pth' % (fold+1)))

    optimizer = RAdam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-5)
    
    os.makedirs('se_resnext50_32x4d', exist_ok=True)

    best_loss = float('inf')
    best_score = 0
    for epoch in range(10):
        print('Epoch [%d/%d]' % (epoch + 1, 10))

        # train for one epoch
        train_loss, train_score = train(
            train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_loss, val_score = validate(val_loader, model, criterion)

        scheduler.step()

        print('loss %.4f - score %.4f - val_loss %.4f - val_score %.4f'
              % (train_loss, train_score, val_loss, val_score))

        if val_loss < best_loss:
            torch.save(model.state_dict(), 'se_resnext50_32x4d/model_%d.pth' %(fold+1))
            best_loss = val_loss
            best_score = val_score
            print("=> saved best model")

    print('val_loss:  %f' % best_loss)
    print('val_score: %f' % best_score)
    
del model
torch.cuda.empty_cache()
get_ipython().system('nvidia-smi')


# ### Inference

# In[ ]:


test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
dir_name = '../input/aptos2019-blindness-detection/test_images'
test_img_paths = dir_name + '/' + test_df['id_code'].values + '.png'
test_labels = np.zeros(len(test_img_paths))

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_set = Dataset(
    test_img_paths,
    test_labels,
    transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=32,
    shuffle=False,
    num_workers=2)

# create model
model = get_model(model_name='se_resnext50_32x4d',
                  num_outputs=1,
                  pretrained=False,
                  freeze_bn=True,
                  dropout_p=0)
model = model.cuda()
model.eval()

probs = []
for fold in range(5):
    print('Fold [%d/%d]' %(fold+1, 5))

    model.load_state_dict(torch.load('se_resnext50_32x4d/model_%d.pth' % (fold+1)))

    probs_fold = []
    with torch.no_grad():
        for i, (input, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
            input = input.cuda()
            output = model(input)

            probs_fold.extend(output.data.cpu().numpy()[:, 0])
    probs_fold = np.array(probs_fold)
    probs.append(probs_fold)

probs = np.mean(probs, axis=0)
l1_probs['se_resnext50_32x4d'] = probs

del model
torch.cuda.empty_cache()
get_ipython().system('nvidia-smi')


# ## SE-ResNeXt101_32x4d

# ### Train

# In[ ]:


aptos2019_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
aptos2019_img_paths = '../input/aptos2019-blindness-detection/train_images/' + aptos2019_df['id_code'].values + '.png'
aptos2019_labels = aptos2019_df['diagnosis'].values

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
test_img_paths = '../input/aptos2019-blindness-detection/test_images/' + test_df['id_code'].values + '.png'
test_labels = 0.4 * pseudo_probs['se_resnext50_32x4d'] + 0.3 * pseudo_probs['se_resnext101_32x4d'] + 0.3 * pseudo_probs['senet154']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
img_paths = []
labels = []
for (train_idx1, val_idx1), (train_idx2, val_idx2) in zip(skf.split(aptos2019_img_paths, aptos2019_labels), kf.split(test_img_paths)):
    img_paths.append((np.hstack((aptos2019_img_paths[train_idx1], test_img_paths[val_idx2])), aptos2019_img_paths[val_idx1]))
    labels.append((np.hstack((aptos2019_labels[train_idx1], test_labels[val_idx2])), aptos2019_labels[val_idx1]))


# In[ ]:


# create model
model = get_model(model_name='se_resnext101_32x4d',
                  num_outputs=1,
                  pretrained=False,
                  freeze_bn=True,
                  dropout_p=0)
model = model.cuda()

criterion = nn.MSELoss().cuda()

best_losses = []
best_scores = []

for fold, ((train_img_paths, val_img_paths), (train_labels, val_labels)) in enumerate(zip(img_paths, labels)):
    print('Fold [%d/%d]' %(fold+1, len(img_paths)))

    # train
    train_set = Dataset(
        train_img_paths,
        train_labels,
        transform=train_transform,
        img_size=288,
        save_img=True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=2)

    val_set = Dataset(
        val_img_paths,
        val_labels,
        transform=val_transform,
        save_img=True)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=2)

    model.load_state_dict(torch.load('../input/se-resnext101-32x4d-081208/model_%d.pth' % (fold+1)))

    optimizer = RAdam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-5)
    
    os.makedirs('se_resnext101_32x4d', exist_ok=True)

    best_loss = float('inf')
    best_score = 0
    for epoch in range(10):
        print('Epoch [%d/%d]' % (epoch + 1, 10))

        # train for one epoch
        train_loss, train_score = train(
            train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_loss, val_score = validate(val_loader, model, criterion)

        scheduler.step()

        print('loss %.4f - score %.4f - val_loss %.4f - val_score %.4f'
              % (train_loss, train_score, val_loss, val_score))

        if val_loss < best_loss:
            torch.save(model.state_dict(), 'se_resnext101_32x4d/model_%d.pth' %(fold+1))
            best_loss = val_loss
            best_score = val_score
            print("=> saved best model")

    print('val_loss:  %f' % best_loss)
    print('val_score: %f' % best_score)
    
del model
torch.cuda.empty_cache()
get_ipython().system('nvidia-smi')


# ### Inference

# In[ ]:


test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
dir_name = '../input/aptos2019-blindness-detection/test_images'
test_img_paths = dir_name + '/' + test_df['id_code'].values + '.png'
test_labels = np.zeros(len(test_img_paths))

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_set = Dataset(
    test_img_paths,
    test_labels,
    transform=test_transform)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=32,
    shuffle=False,
    num_workers=2)

# create model
model = get_model(model_name='se_resnext101_32x4d',
                  num_outputs=1,
                  pretrained=False,
                  freeze_bn=True,
                  dropout_p=0)
model = model.cuda()
model.eval()

probs = []
for fold in range(5):
    print('Fold [%d/%d]' %(fold+1, 5))

    model.load_state_dict(torch.load('se_resnext101_32x4d/model_%d.pth' % (fold+1)))

    probs_fold = []
    with torch.no_grad():
        for i, (input, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
            input = input.cuda()
            output = model(input)

            probs_fold.extend(output.data.cpu().numpy()[:, 0])
    probs_fold = np.array(probs_fold)
    probs.append(probs_fold)

probs = np.mean(probs, axis=0)
l1_probs['se_resnext101_32x4d'] = probs

del model
torch.cuda.empty_cache()
get_ipython().system('nvidia-smi')


# # Ensemble

# In[ ]:


preds = 0.5 * l1_probs['se_resnext50_32x4d'] + 0.5 * l1_probs['se_resnext101_32x4d']


# In[ ]:


thrs = [0.5, 1.5, 2.5, 3.5]
preds[preds < thrs[0]] = 0
preds[(preds >= thrs[0]) & (preds < thrs[1])] = 1
preds[(preds >= thrs[1]) & (preds < thrs[2])] = 2
preds[(preds >= thrs[2]) & (preds < thrs[3])] = 3
preds[preds >= thrs[3]] = 4
preds = preds.astype('int')

test_df['diagnosis'] = preds
test_df.to_csv('submission.csv', index=False)


# In[ ]:


get_ipython().system('rm processed/*')

