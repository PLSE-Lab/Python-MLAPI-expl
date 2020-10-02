#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install torchsummary')


# # Libraries

# In[ ]:


import os
import random
from collections import defaultdict
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

import numpy as np
import pandas as pd
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

from fastprogress import master_bar, progress_bar
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import make_grid
from torchsummary import summary


get_ipython().run_line_magic('matplotlib', 'inline')


# ## utils

# In[ ]:


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def get_logger(
    filename='log',
    disable_stream_handler=False,
    disable_file_handler=False
):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    
    if not disable_stream_handler:
        handler1 = StreamHandler()
        handler1.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler1)
    
    if not disable_file_handler:
        handler2 = FileHandler(filename=f"{filename}.log")
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler2)
    
    return logger


# In[ ]:


def get_image_status(train_path, test_path, use_tqdm=False):
    if use_tqdm:
        train_path = tqdm(train_path)
        test_path = tqdm(test_path)

    rgb_status = {'R': defaultdict(list), 'G': defaultdict(list), 'B': defaultdict(list)}
    hsv_status = {'H': defaultdict(list), 'S': defaultdict(list), 'V': defaultdict(list)}
    size_status = {'height': defaultdict(list), 'width': defaultdict(list), 'aspect': defaultdict(list)}
    
    def get_stats_dict(path):
        rgb_status = {'R': defaultdict(list), 'G': defaultdict(list), 'B': defaultdict(list)}
        hsv_status = {'H': defaultdict(list), 'S': defaultdict(list), 'V': defaultdict(list)}
        size_status = defaultdict(list)
        
        for p in path:
            img = cv2.imread(p)
            for i, c in enumerate(['B', 'G', 'R']):
                rgb_status[c]['mean'].append(img[:,:,i].mean())
                rgb_status[c]['std'].append(img[:,:,i].std())
                rgb_status[c]['max'].append(img[:,:,i].max())
                rgb_status[c]['min'].append(img[:,:,i].min())

            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            for i, c in enumerate(['H', 'S', 'V']):
                hsv_status[c]['mean'].append(img[:,:,i].mean())
                hsv_status[c]['std'].append(img[:,:,i].std())
                hsv_status[c]['max'].append(img[:,:,i].max())
                hsv_status[c]['min'].append(img[:,:,i].min())
        
            size_status['height'].append(img.shape[0])
            size_status['width'].append(img.shape[1])
            size_status['aspect'].append(img.shape[0] / img.shape[1])
        
        return rgb_status, hsv_status, size_status
    
    print('loading train images...')
    tr_rgb_status, tr_hsv_status, tr_size_status = get_stats_dict(train_path)
    
    print('loading test images...')
    te_rgb_status, te_hsv_status, te_size_status = get_stats_dict(test_path)
    
    return [tr_rgb_status, tr_hsv_status, tr_size_status], [te_rgb_status, te_hsv_status, te_size_status]


def show_image_status(tr_rgb_status, tr_hsv_status, tr_size_status, 
                      te_rgb_status, te_hsv_status, te_size_status,
                      fig_height=3, fig_width=5):

    print('** RGB status histogram (R/G/B)-(mean/std/max/min) **')
    fig, axes = plt.subplots(3, 4)
    fig.set_size_inches(4 * fig_width, 3 * fig_height)

    sns.distplot(tr_rgb_status['R']['mean'], ax=axes[0, 0], kde=False)
    sns.distplot(te_rgb_status['R']['mean'], ax=axes[0, 0], kde=False)
    sns.distplot(tr_rgb_status['R']['std'], ax=axes[0, 1], kde=False)
    sns.distplot(te_rgb_status['R']['std'], ax=axes[0, 1], kde=False)
    sns.distplot(tr_rgb_status['R']['max'], ax=axes[0, 2], kde=False)
    sns.distplot(te_rgb_status['R']['max'], ax=axes[0, 2], kde=False)
    sns.distplot(tr_rgb_status['R']['min'], ax=axes[0, 3], kde=False)
    sns.distplot(te_rgb_status['R']['min'], ax=axes[0, 3], kde=False)

    sns.distplot(tr_rgb_status['G']['mean'], ax=axes[1, 0], kde=False)
    sns.distplot(te_rgb_status['G']['mean'], ax=axes[1, 0], kde=False)
    sns.distplot(tr_rgb_status['G']['std'], ax=axes[1, 1], kde=False)
    sns.distplot(te_rgb_status['G']['std'], ax=axes[1, 1], kde=False)
    sns.distplot(tr_rgb_status['G']['max'], ax=axes[1, 2], kde=False)
    sns.distplot(te_rgb_status['G']['max'], ax=axes[1, 2], kde=False)
    sns.distplot(tr_rgb_status['G']['min'], ax=axes[1, 3], kde=False)
    sns.distplot(te_rgb_status['G']['min'], ax=axes[1, 3], kde=False)

    sns.distplot(tr_rgb_status['B']['mean'], ax=axes[2, 0], kde=False)
    sns.distplot(te_rgb_status['B']['mean'], ax=axes[2, 0], kde=False)
    sns.distplot(tr_rgb_status['B']['std'], ax=axes[2, 1], kde=False)
    sns.distplot(te_rgb_status['B']['std'], ax=axes[2, 1], kde=False)
    sns.distplot(tr_rgb_status['B']['max'], ax=axes[2, 2], kde=False)
    sns.distplot(te_rgb_status['B']['max'], ax=axes[2, 2], kde=False)
    sns.distplot(tr_rgb_status['B']['min'], ax=axes[2, 3], kde=False)
    sns.distplot(te_rgb_status['B']['min'], ax=axes[2, 3], kde=False)
    
    plt.show()
    
    print('** HSV status histogram (H/S/V)-(mean/std/max/min) **')
    fig, axes = plt.subplots(3, 4)
    fig.set_size_inches(4 * fig_width, 3 * fig_height)

    sns.distplot(tr_hsv_status['H']['mean'], ax=axes[0, 0], kde=False)
    sns.distplot(te_hsv_status['H']['mean'], ax=axes[0, 0], kde=False)
    sns.distplot(tr_hsv_status['H']['std'], ax=axes[0, 1], kde=False)
    sns.distplot(te_hsv_status['H']['std'], ax=axes[0, 1], kde=False)
    sns.distplot(tr_hsv_status['H']['max'], ax=axes[0, 2], kde=False)
    sns.distplot(te_hsv_status['H']['max'], ax=axes[0, 2], kde=False)
    sns.distplot(tr_hsv_status['H']['min'], ax=axes[0, 3], kde=False)
    sns.distplot(te_hsv_status['H']['min'], ax=axes[0, 3], kde=False)

    sns.distplot(tr_hsv_status['S']['mean'], ax=axes[1, 0], kde=False)
    sns.distplot(te_hsv_status['S']['mean'], ax=axes[1, 0], kde=False)
    sns.distplot(tr_hsv_status['S']['std'], ax=axes[1, 1], kde=False)
    sns.distplot(te_hsv_status['S']['std'], ax=axes[1, 1], kde=False)
    sns.distplot(tr_hsv_status['S']['max'], ax=axes[1, 2], kde=False)
    sns.distplot(te_hsv_status['S']['max'], ax=axes[1, 2], kde=False)
    sns.distplot(tr_hsv_status['S']['min'], ax=axes[1, 3], kde=False)
    sns.distplot(te_hsv_status['S']['min'], ax=axes[1, 3], kde=False)

    sns.distplot(tr_hsv_status['V']['mean'], ax=axes[2, 0], kde=False)
    sns.distplot(te_hsv_status['V']['mean'], ax=axes[2, 0], kde=False)
    sns.distplot(tr_hsv_status['V']['std'], ax=axes[2, 1], kde=False)
    sns.distplot(te_hsv_status['V']['std'], ax=axes[2, 1], kde=False)
    sns.distplot(tr_hsv_status['V']['max'], ax=axes[2, 2], kde=False)
    sns.distplot(te_hsv_status['V']['max'], ax=axes[2, 2], kde=False)
    sns.distplot(tr_hsv_status['V']['min'], ax=axes[2, 3], kde=False)
    sns.distplot(te_hsv_status['V']['min'], ax=axes[2, 3], kde=False)
    
    plt.show()
    
    print('** Height/Width/Aspect status histogram **')
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(3 * fig_width, fig_height)

    sns.distplot(tr_size_status['height'], ax=axes[0], kde=False)
    sns.distplot(te_size_status['height'], ax=axes[0], kde=False)
    
    sns.distplot(tr_size_status['width'], ax=axes[1], kde=False)
    sns.distplot(te_size_status['width'], ax=axes[1], kde=False)
    
    sns.distplot(tr_size_status['aspect'], ax=axes[2], kde=False)
    sns.distplot(te_size_status['aspect'], ax=axes[2], kde=False)
    
    plt.show()


# # Loading

# In[ ]:


logger = get_logger(
    filename='running',
    disable_stream_handler=False,
    disable_file_handler=False,
)


# In[ ]:


INPUT_DIR = '../input/ailab-ml-training-1/'
ARTIFACT_DIR = '../input/v1-ailab1-cv/'

PATH = {
    'train': os.path.join(INPUT_DIR, 'train.csv'),
    'submission': os.path.join(INPUT_DIR, 'sample_submission.csv'),
    'train_image_dir': os.path.join(INPUT_DIR, 'train_images/train_images'),
    'test_image_dir': os.path.join(INPUT_DIR, 'test_images/test_images'),
    'state_dict': os.path.join(ARTIFACT_DIR, 'best_state_dicts.pth'),
    'oof': os.path.join(ARTIFACT_DIR, 'oof.npy'),
    'predictions': os.path.join(ARTIFACT_DIR, 'predictions.npy'),
}

ID = 'fname'
TARGET = 'label'

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(SEED)


# In[ ]:


train_df = pd.read_csv(PATH['train'])
submission_df = pd.read_csv(PATH['submission'])


# In[ ]:


train_df[ID] = train_df[ID].apply(lambda x: os.path.join(PATH['train_image_dir'], x))


# # Pseudo Labeling

# In[ ]:


def softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)


# In[ ]:


predictions = np.load(PATH['predictions'])
predictions = softmax(predictions)
predictions_label, predictions_proba = np.argmax(predictions, axis=1), np.max(predictions, axis=1)

pseudo_df = submission_df.copy()
pseudo_df[TARGET] = predictions_label
pseudo_df['proba'] = predictions_proba

pseudo_df = pseudo_df.loc[pseudo_df['proba'] > 0.999, :]
pseudo_df = pseudo_df.drop('proba', axis=1)
pseudo_df[ID] = pseudo_df[ID].apply(lambda x: os.path.join(PATH['test_image_dir'], x))


# In[ ]:


train_df.shape[0], pseudo_df.shape[0], submission_df.shape[0]


# # EDA

# In[ ]:


# print(f'number of train data: {len(train_df)}')
# print(f'number of test data: {len(submission_df)}')


# In[ ]:


# print(f'number of unique label: {train_df[TARGET].nunique()}')


# In[ ]:


# sns.countplot(train_df[TARGET])
# plt.title('train label distribution')
# plt.show()


# In[ ]:


# train_path = [os.path.join(PATH['train_image_dir'], fn) for fn in train_df[ID]]
# test_path = [os.path.join(PATH['test_image_dir'], fn) for fn in submission_df[ID]]
# [train_rgb_status, _, _], [test_rgb_status, _, _] = get_image_status(train_path, test_path, use_tqdm=False)


# In[ ]:


# print('Mean/std of color mean')
# print('train: {:.5f} (+-) {:.3f}'.format(
#     np.mean(train_rgb_status['R']['mean']),
#     np.mean(train_rgb_status['R']['std'])))
# print('test : {:.5f} (+-) {:.3f}'.format(
#     np.mean(test_rgb_status['R']['mean']),
#     np.mean(test_rgb_status['R']['std'])))


# ```
# Mean/std of color mean
# train: 48.89935 (+-) 86.269
# test : 47.06204 (+-) 84.471
# ```

# ## dataset, dataloader

# In[ ]:


class KmnistDataset(Dataset):
    def __init__(
        self,
        paths,
        labels,
        transform=None,
        with_memory_cache=False,
    ):
        super().__init__()

        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.with_memory_cache = with_memory_cache

        if with_memory_cache:
            self.images = [None,] * len(paths)
    
    def load_image(self, path):
#         image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         image = image.reshape(image.shape[0], image.shape[1], 1)
        image = cv2.imread(path)
        
        return image
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        if self.with_memory_cache:
            image = self.images[idx]
            if image is None:
                path = self.paths[idx]
                image = self.load_image(path)
                self.images[idx] = image
        else:
            path = self.paths[idx]
            image = self.load_image(path)
            
        label = self.labels[idx]
        
        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        return image, label


# In[ ]:


def get_dataloader(
    X,
    Y,
    transform=None,
    with_memory_cache=False,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
):
    dataset = KmnistDataset(
        X,
        Y,
        transform=transform,
        with_memory_cache=with_memory_cache,
    )
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return loader


# In[ ]:


num_samples = 20
sample = train_df.groupby(TARGET).apply(lambda df: df.sample(num_samples))
fnames = sample[ID].to_list()
labels = sample[TARGET].to_list()


# In[ ]:


# transform = A.Compose([
#     A.Normalize((0,), (1,)),
#     ToTensorV2(),
# ])

# loader = get_dataloader(
#     fnames, labels, PATH['train_image_dir'], 
#     transform=transform, batch_size=len(fnames))
# loader = iter(loader)
# x, y = next(loader)

# imgtile = make_grid(x, nrow=num_samples)
# imgtile = imgtile.permute(1, 2, 0).cpu()
# plt.figure(figsize=(2 * 10, 2 * num_samples))
# plt.imshow(imgtile)
# plt.show()


# In[ ]:


IMSIZE = 128

transform = A.Compose([
    A.Rotate(limit=5, p=1.0),
    A.RandomResizedCrop(IMSIZE, IMSIZE, scale=(0.9, 1.0), ratio=(0.9, 1.1), p=1.0),
    A.Cutout(num_holes=1, max_h_size=IMSIZE//2, max_w_size=IMSIZE//2, fill_value=48.89935, p=1.0),
    A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ToTensorV2(),
])

loader = get_dataloader(fnames, labels, transform=transform, batch_size=len(fnames))
loader = iter(loader)
x, y = next(loader)

imgtile = make_grid(x, nrow=num_samples, normalize=True)
imgtile = imgtile.permute(1, 2, 0).cpu()
plt.figure(figsize=(2 * 10, 2 * num_samples))
plt.imshow(imgtile)
plt.show()


# ## model

# In[ ]:


# https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.htm


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=padding, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    
class SSELayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Sequential(
            conv1x1(channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv(x)
        return x * y.expand_as(x)


class CSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    
class SCSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.sse = SSELayer(channel)
        self.cse = CSELayer(channel, reduction)
    
    def forward(self, x):
        return self.sse(x) + self.cse(x)
    

def se_layer(channel, se_type=None, reduction=16):
    if se_type is None:
        return nn.Identity()
    elif se_type == 'cse':
        return CSELayer(channel, reduction)
    elif se_type == 'sse':
        return SSELayer(channel)
    elif se_type == 'scse':
        return SCSELayer(channel, reduction)
    else:
        raise NotImplementedError
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=None, se_type=None, reduction=16):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.se = se_layer(planes, se_type=se_type, reduction=reduction)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, width_per_group=64,
                 se_type=None, reduction=16):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.base_width = width_per_group
        self.se_type = se_type
        self.reduction = reduction
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, norm_layer=norm_layer,
                            se_type=self.se_type, reduction=self.reduction))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                se_type=self.se_type, reduction=self.reduction))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
    
def small_resnet18(se_type='scse'):
    return ResNet(BasicBlock, [2, 2, 2, 2], se_type=se_type)


def small_resnet34(se_type='scse'):
    return ResNet(BasicBlock, [3, 4, 6, 3], se_type=se_type)


# In[ ]:


# class Model(nn.Module):
#     def __init__(self, depth=18, se_type='scse'):
#         super().__init__()
#         if depth == 18:
#             self.model = small_resnet18(se_type=se_type)
#         elif depth == 34:
#             self.model = small_resnet34(se_type=se_type)
#         else:
#             raise NotImplementedError
    
#     def forward(self, x):
#         return self.model(x)


# class Model(nn.Module):
#     def __init__(self, depth=34):
#         super().__init__()
#         if depth == 18:
#             self.model = models.resnet18(pretrained=True)
#             self.model.fc = nn.Linear(self.model.fc.in_features, 10)
#         elif depth == 34:
#             self.model = models.resnet34(pretrained=True)
#             self.model.fc = nn.Linear(self.model.fc.in_features, 10)
#         else:
#             raise NotImplementedError
    
#     def forward(self, x):
#         return self.model(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
    
    def forward(self, x):
        return self.model(x)


# In[ ]:


model = Model()
summary(model, (3, IMSIZE, IMSIZE), device='cpu')


# # Training

# In[ ]:


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_score_torch(y_pred, y):
    y_pred = torch.argmax(y_pred, axis=1).cpu().numpy()
    y = y.cpu().numpy()

    return accuracy_score(y, y_pred)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# In[ ]:


def train(
    params,
    model,
    optimizer,
    criterion,
    dataloader,
    parent_bar=None,
):
    model.train()
    losses = AverageMeter()
    metrics = AverageMeter()
    
    for x, y in progress_bar(dataloader, parent=parent_bar):
        x = x.to(dtype=torch.float32, device=DEVICE)
        y = y.to(dtype=torch.long, device=DEVICE)
        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item())
        metrics.update(accuracy_score_torch(y_pred, y))
    
    return losses.avg, metrics.avg


def valid(
    params,
    model,
    criterion,
    dataloader,
):
    model.eval()
    losses = AverageMeter()
    metrics = AverageMeter()

    for x, y in dataloader:
        x = x.to(dtype=torch.float32, device=DEVICE)
        y = y.to(dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            y_pred = model(x)
            loss = criterion(y_pred, y)
        
        losses.update(loss.item())
        metrics.update(accuracy_score_torch(y_pred, y))
    
    return losses.avg, metrics.avg


# In[ ]:


params = {
    'n_splits': 5,
    'batch_size': 64,
    'test_batch_size': 128,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'epochs': 25,
    'restart': 25,
}


# In[ ]:


kf = StratifiedKFold(n_splits=params['n_splits'], random_state=SEED, shuffle=True)
oof = np.zeros((len(train_df), 10))
best_state_dicts = []

logger.info('** start kfold training **')
for i, (dev_idx, val_idx) in enumerate(kf.split(train_df[ID], train_df[TARGET])):
    logger.info(f'[fold: {i}]')
    
    # ------------------------------
    # dev/val dataloader
    # ------------------------------
    
    dev_df = train_df.iloc[dev_idx, :].reset_index(drop=True)
    val_df = train_df.iloc[val_idx, :].reset_index(drop=True)
    
    # pseudo labeling
    dev_df = pd.concat([dev_df, pseudo_df], axis=0).reset_index(drop=True)
    
    dev_transform = A.Compose([
        A.Rotate(limit=5, p=1.0),
        A.RandomResizedCrop(IMSIZE, IMSIZE, scale=(0.9, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.Cutout(num_holes=1, max_h_size=IMSIZE//2, max_w_size=IMSIZE//2, fill_value=48.89935, p=1.0),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(IMSIZE, IMSIZE, p=1),
        A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    dev_dataloader = get_dataloader(
        dev_df[ID],
        dev_df[TARGET],
        transform=dev_transform,
        with_memory_cache=True,
        batch_size=params['batch_size'],
        shuffle=True,
        pin_memory=True,
    )
    val_dataloader = get_dataloader(
        val_df[ID],
        val_df[TARGET],
        transform=val_transform,
        with_memory_cache=True,
        batch_size=params['test_batch_size'],
        shuffle=False,
        pin_memory=True,
    )
    
    # ----------------------------------------
    # model, loss, optimizer, scheduler
    # ----------------------------------------

    model = Model().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, params['restart'])
    criterion = nn.CrossEntropyLoss()
    
    # ------------------------------
    # training loop
    # ------------------------------
    
    best_loss = np.inf
    best_state_dict = model.state_dict()
    
    mb = master_bar(range(params['epochs']))
    for epoch in mb:
        # train
        dev_loss, dev_metric = train(
            params, model, optimizer, criterion, dev_dataloader, parent_bar=mb)
        
        # valid
        val_loss, val_metric = valid(
            params, model, criterion, val_dataloader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state_dict = model.state_dict()

        msg = (
            'epoch: {}/{}'
            ' - lr: {:.6f}'
            ' - loss: {:.5f}'
            ' - acc: {:.5f}'
            ' - val_loss: {:.5f}'
            ' - val_acc: {:.5f}'
        ).format(
            epoch + 1,
            params['epochs'],
            optimizer.state_dict()['param_groups'][0]['lr'],
            dev_loss,
            dev_metric,
            val_loss,
            val_metric,
        )
        logger.info(msg)

        if scheduler is not None:
            scheduler.step()
    
    # ------------------------------
    # end of training loop
    # ------------------------------
    
    best_state_dicts.append(best_state_dict)

    model.load_state_dict(best_state_dict)
    model.eval()
    _oof = []

    for x, _ in val_dataloader:
        x = x.to(dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            y_pred = model(x)
            _oof.append(y_pred.cpu().numpy())
    
    oof[val_idx] = np.concatenate(_oof)


# In[ ]:


cv_score = accuracy_score(train_df[TARGET], np.argmax(oof, axis=1))
logger.info(f'CV: {cv_score:.5f}')


# # Prediction

# In[ ]:


test_transform = A.Compose([
    A.Resize(IMSIZE, IMSIZE, p=1),
    A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ToTensorV2(),
])

test_dataloader = get_dataloader(
    submission_df[ID].apply(lambda x: os.path.join(PATH['test_image_dir'], x)),
    submission_df[TARGET],
    transform=test_transform,
    with_memory_cache=True,
    batch_size=params['test_batch_size'],
    shuffle=False,
    pin_memory=True,
)


# In[ ]:


predictions = np.zeros((len(submission_df), 10))

for state_dict in best_state_dicts:
    model = Model().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    _predictions = []
    
    for x, _ in test_dataloader:
        x = x.to(dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            y_pred = model(x)
            _predictions.append(y_pred.cpu().numpy())
    
    _predictions = np.concatenate(_predictions)
    predictions += _predictions / len(best_state_dicts)


# # Save outputs

# In[ ]:


np.save('oof', oof)
np.save('predictions', predictions)

saved_best_state_dicts = {}
for i, bsd in enumerate(best_state_dicts):
    saved_best_state_dicts[f'f{i}'] = bsd
torch.save(saved_best_state_dicts, 'best_state_dicts.pth')

submission_df[TARGET] = np.argmax(predictions, axis=1).tolist()


# In[ ]:


submission_df.to_csv('submission.csv', index=False)
from IPython.display import FileLink
FileLink('submission.csv')


# In[ ]:




