#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys, math, time
import pandas as pd
import numpy as np
from glob import glob

from PIL import Image
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook, tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


INPUT_SIZE = (300,300)
data = pd.read_csv('../input/imet-2019-fgvc6/train.csv')


# In[ ]:


# unique labels - for sigmoid predict (1103 class)
labels_count = pd.Series(data['attribute_ids'].str.split(' ').sum()).value_counts()
pd.DataFrame(
    data=np.array([labels_count.keys().values, labels_count.values]).T,
    columns=['label', 'count'])[:3]


# In[ ]:


# unique labels group - for softmax predict (9280 class)
count_groups = data['attribute_ids'].value_counts()
pd.DataFrame(
    data=np.array([count_groups.keys().values, count_groups.values]).T,
    columns=['group labels', 'count'])[:3]


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
torch.manual_seed(42)


# In[ ]:


class train_dataset(Dataset):
    def __init__(self, df, path_to_images, shape=INPUT_SIZE, augmentation=True):
        self.df = df
        self.shape = shape
        self.augmentation = augmentation
        self.path_to_images = path_to_images
        self.labels_group_encoder = {labels:0 
                                     for labels in df['attribute_ids'].unique()}
        labels_groups_mask = df['attribute_ids'].value_counts() > 1
        labels_group = df['attribute_ids'].value_counts()[labels_groups_mask]
        for idx, labels_group in enumerate(labels_group.index):
            self.labels_group_encoder[labels_group] = idx
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        path_to_image = os.path.join(self.path_to_images, 
                                     self.df.iloc[idx]['id'] + '.png')
        image = self._load_image(path_to_image)
        
        # create labels for sigmoid output
        raw_sigmoid_label = self.df.iloc[idx]['attribute_ids'].strip().split(' ')
        sigmoid_labels = torch.zeros(1103)
        sigmoid_labels[[int(e) for e in raw_sigmoid_label]] = 1
        
        # create labels for softmax output
        class_number = self.labels_group_encoder[self.df.iloc[idx]['attribute_ids']]
        softmax_label = torch.tensor(class_number).long()
        return (image, sigmoid_labels, softmax_label)
    
    def _load_image(self, path_to_image):
        image = np.array(Image.open(path_to_image))
        if self.augmentation:
            image = self._augumentation_image(image)
        image = cv2.resize(image, (self.shape[0], self.shape[1]))
        image = np.divide(image, 255)
        image = image.transpose(2,0,1)
        return torch.tensor(image, dtype=torch.float)
    
    def _augumentation_image(self, image): 
        sometimes = lambda aug: iaa.Sometimes(0.1, aug)
        
        angle = np.random.randint(360)
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Crop(px=(0, 50)),
            sometimes(
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-25, 25),
                    shear=(-16, 16),
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL)),
            sometimes(iaa.GaussianBlur(sigma=(0, 2.0))),
            sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25))
        ], random_order=True)
        return augment_img.augment_image(image)


# In[ ]:


from torchvision.models.resnet import conv3x3
from torchvision.models.resnet import conv1x1

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

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


class ResNet(nn.Module):

    def __init__(self, block, layers, sigmoid_num_class=1103, softmax_num_class=9280, 
                 zero_init_residual=False, groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 
                                       norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, 
                                       norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, 
                                       norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, 
                                       norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid_fc = nn.Linear(512 * block.expansion, sigmoid_num_class)
        self.softmax_fc = nn.Linear(512 * block.expansion, softmax_num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, 
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        sigmoid_output = self.sigmoid_fc(x) # sigmoid output branch
        softmax_output = self.softmax_fc(x) # softmax output branch
        
        return sigmoid_output, softmax_output


# In[ ]:


def fbeta_score(prob, label, threshold=0.5, beta=2):
    prob = prob > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    fscore = (1+beta**2)*precision*recall/(beta**2*precision+recall+1e-12)
    return fscore.mean(0)


# In[ ]:


def train_epoch(model, train_loader, sigmoid_criterion, softmax_criterion, loss_blender, optimizer):
    model.train()
    
    history = []
    with tqdm(total=len(train_loader), file=sys.stdout) as pbar:
        for step, (features, sigmoid_labels, softmax_labels) in enumerate(train_loader):
            features, sigmoid_labels, softmax_labels =             features.cuda(), sigmoid_labels.cuda(), softmax_labels.cuda()
            optimizer.zero_grad()
            
            sigmoid_logits, softmax_logits = model(features)
            sigmoid_loss = sigmoid_criterion(sigmoid_logits, sigmoid_labels)
            softmax_loss = softmax_criterion(softmax_logits, softmax_labels)
            total_loss = loss_blender(sigmoid_loss, softmax_loss)
            
            total_loss.backward()
            optimizer.step()

            if step % 100 == 0:
                f2 = fbeta_score(sigmoid_labels, torch.sigmoid(sigmoid_logits))
                history.append({
                    'f2_socre': f2,
                    'total_loss':total_loss.item(),
                    'softmax_loss':softmax_loss.item(),
                    'sigmoid_loss':sigmoid_loss.item()})
                pbar.set_description(
                    'f2 (th=0.5): {0:.4}, total_loss {1:.4}, softmax_loss {2:.4}, sigmoid_loss {3:.4}'.format(
                        f2.item(), total_loss.item(), softmax_loss.item(), sigmoid_loss.item()))
                pbar.update(100)
        pbar.close()
        return history


# In[ ]:


def model_eval(model, valid_loader, sigmoid_criterion, softmax_criterion, loss_blender):
    with torch.no_grad():
        model.eval()

        total_eval = 0; total_sigmoid_loss = 0; total_softmax_loss = 0; total_loss = 0
        for step, (features, sigmoid_labels, softmax_labels) in enumerate(valid_loader):
            features, sigmoid_labels, softmax_labels =             features.cuda(), sigmoid_labels.cuda(), softmax_labels.cuda()
            
            sigmoid_logits, softmax_logits = model(features)
            
            sigmoid_loss = sigmoid_criterion(sigmoid_logits, sigmoid_labels)
            softmax_loss = softmax_criterion(softmax_logits, softmax_labels)
            
            total_loss += loss_blender(sigmoid_loss, softmax_loss)
            total_eval += fbeta_score(sigmoid_labels, torch.sigmoid(sigmoid_logits))
            
            total_sigmoid_loss += sigmoid_loss.item()
            total_softmax_loss += softmax_loss.item()

        return total_loss/len(valid_loader), total_sigmoid_loss/len(valid_loader), softmax_loss/len(valid_loader), total_eval/len(valid_loader)


# In[ ]:


def model_train(model, train_loader, valid_loader, sigmoid_criterion, softmax_criterion, loss_blender,
                optimizer, n_epoch, checkpoint_path='./model.pth'):
    
    train_history = []; best_score = 0
    for epoch in range(n_epoch):
        train_history += train_epoch(
            model, train_loader, sigmoid_criterion, softmax_criterion, loss_blender, optimizer)
        val_total_loss, val_sigmoid_loss, val_softmax_loss, valid_score = model_eval(
            model, valid_loader, sigmoid_criterion, softmax_criterion, loss_blender)
        
        print('val loss: {0:.4}, val f2 score: {1:.4}'.format(
            val_total_loss, valid_score))
        if valid_score > best_score:
            print('save')
            torch.save(model.state_dict(), checkpoint_path)
            best_score = valid_score

    return train_history


# In[ ]:


def show_history(history, figsize=(10,4)):
    plt.figure(figsize=figsize)
    total_loss = plt.plot([e['total_loss'] for e in history])
    softmax_loss = plt.plot([e['softmax_loss'] for e in history])
    sigmoid_loss = plt.plot([e['sigmoid_loss'] for e in history])
    plt.legend((total_loss[0], softmax_loss[0], sigmoid_loss[0]), 
               ('total_loss', 'softmax_loss', 'sigmoid_loss'), fontsize=12)


# In[ ]:


from sklearn.model_selection import train_test_split
train_idx, valid_idx, train_targets, valid_target = train_test_split(
    np.arange(data.shape[0]), data['attribute_ids'], test_size=0.1, random_state=42)


# In[ ]:


trainloader = DataLoader(
    train_dataset(data.iloc[train_idx], '../input/imet-2019-fgvc6/train/', augmentation=True), 
    batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

validloader = DataLoader(
    train_dataset(data.iloc[valid_idx], '../input/imet-2019-fgvc6/train/', augmentation=False), 
    batch_size=32, shuffle=False, num_workers=4, pin_memory=True)


# In[ ]:


donor = models.resnet18()
donor.load_state_dict(torch.load('../input/resnet18/resnet18.pth'))

# create empty model-acceptor
model = ResNet(BasicBlock, [2,2,2,2])

# transfer weights
model.layer1.load_state_dict(donor.layer1.state_dict())
model.layer2.load_state_dict(donor.layer2.state_dict())
model.layer3.load_state_dict(donor.layer3.state_dict())
model.layer4.load_state_dict(donor.layer4.state_dict())


# In[ ]:


history = model_train(
    model.cuda(), trainloader, validloader, 
    sigmoid_criterion=nn.BCEWithLogitsLoss(), 
    softmax_criterion=nn.CrossEntropyLoss(),
    loss_blender=lambda sigmoid_loss, softmax_loss: 0.99*sigmoid_loss + 0.01*softmax_loss,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.0002),
    n_epoch=25)


# In[ ]:


show_history(history)


# In[ ]:


scores = []
with torch.no_grad():
    model.eval()
    for step, (features, sigmoid_labels, softmax_labels) in enumerate(validloader):
        features, sigmoid_labels, softmax_labels =        features.cuda(), sigmoid_labels.cuda(), softmax_labels.cuda()
        
        sigmoid_logits, softmax_logits = model(features)
        batch_scores = np.array([fbeta_score(torch.sigmoid(sigmoid_logits), sigmoid_labels, threshold=conf) 
                                 for conf in np.arange(0,1,0.01)])
        scores.append(batch_scores)
        
mean_scores = torch.FloatTensor(scores).cpu().numpy()

plt.plot(np.arange(0,1,0.01), mean_scores.mean(axis=0), linewidth=2, color='r')
sigmoid_best_score = mean_scores.mean(axis=0)[mean_scores.mean(axis=0).argmax()]
sigmoid_best_conf = np.arange(0,1,0.01)[mean_scores.mean(axis=0).argmax()]
plt.plot(sigmoid_best_conf, sigmoid_best_score, color='g', marker='x')
print('best score', sigmoid_best_score)
print('best conf', sigmoid_best_conf)


# In[ ]:


td = train_dataset(data.iloc[valid_idx], '../input/imet-2019-fgvc6/train/')
softmax_label_decoder = {v:k for (k,v) in zip(
    td.labels_group_encoder.keys(), td.labels_group_encoder.values())}


# In[ ]:


submit = pd.read_csv('../input/imet-2019-fgvc6/sample_submission.csv')

predicts = []; softmax_min_conf = 0.99
with torch.no_grad():
    for fpath in tqdm_notebook(submit['id']):
        image = np.array(Image.open('../input/imet-2019-fgvc6/test/'+fpath+'.png'))
        image = cv2.resize(image, INPUT_SIZE)
        image = np.divide(image, 255)
        image = image.transpose(2,0,1)
        image = torch.tensor(image[np.newaxis], dtype=torch.float)
        image = image.cuda()
        sigmoid_logits, softmax_logits = model(image)
        
        softmax_predicts = torch.softmax(softmax_logits, dim=1)[0]
        idx = softmax_predicts.argmax()
        softmax_class_number = int(idx.cpu().numpy())
        
        # if softmax have hight conf - softmax output predict, else sigmoid
        if softmax_predicts[idx] > softmax_min_conf and softmax_class_number in softmax_label_decoder.keys():
            predicts.append(softmax_label_decoder[softmax_class_number])
        else:
            sigmoid_predict = torch.sigmoid(sigmoid_logits).cpu().numpy()[0]
            sigmoid_predict = np.arange(len(sigmoid_predict))[sigmoid_predict > sigmoid_best_conf]
            predicts.append(str.join(' ', [str(e) for e in sigmoid_predict]))
        
submit['attribute_ids'] = predicts
submit.to_csv('./submission.csv', index=False)

