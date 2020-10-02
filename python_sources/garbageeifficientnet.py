#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install torch==1.1.0
#!pip install pretrainedmodels
#!pip install pytorchcv
#!pip install albumentations
get_ipython().system('pip install efficientnet_pytorch')
from efficientnet_pytorch import EfficientNet
#from pretrainedmodels import se_resnext101_32x4d
#from pytorchcv.model_provider import get_model as ptcv_get_model
#import albumentations
#from albumentations import torch as AT


# In[ ]:


get_ipython().system('git clone https://github.com/NVIDIA/apex')
get_ipython().system('pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex')


# In[ ]:


import cv2
import matplotlib.pyplot as plt
from os.path import isfile
import torch.nn.init as init
import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
import os
from PIL import Image, ImageFilter
#print(os.listdir("../input"))
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop
import time
from torch.autograd import Variable
import torch.functional as F
from tqdm import tqdm
from sklearn import metrics
import urllib
import pickle
import cv2
import torch.nn.functional as F
from torchvision import models
import seaborn as sns
import random
from PIL import Image
import sys
from apex import amp


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[ ]:


#train = '../input/hgarbage/garbage_classify/train_data/'
#train_csv = pd.read_csv("../input/hgarbage/garbage_classify/img_info.csv")

#train_df, val_df = train_test_split(train_csv, test_size=0.1, random_state=24, stratify=train_csv.cls)
#train_df.reset_index(drop=True, inplace=True)
#val_df.reset_index(drop=True, inplace=True)
#num_cls = len(train_csv['cls'].unique())


# In[ ]:


train= "../input/hwdataset/my_data/my_data/"
trn_csv = pd.read_csv("../input/hwdataset/train.csv")
trn_csv = trn_csv[:14803]
fold = 4
train_df, val_df = train_test_split(trn_csv, test_size=0.1, random_state=24, stratify=trn_csv.cls)
#trn_idx = list(StratifiedKFold(n_splits=5, random_state=24, shuffle=True).split(trn_csv, trn_csv.cls))[fold][0]
#val_idx = list(StratifiedKFold(n_splits=5, random_state=24, shuffle=True).split(trn_csv, trn_csv.cls))[fold][1]
#train_df = trn_csv.iloc[trn_idx]
#val_df = trn_csv.iloc[val_idx]
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)


# In[ ]:


groupDataFrame=train_df.groupby(by=['cls'])
labels=groupDataFrame.size()
print("length of label is ",len(labels))
maxNum=max(labels)
lst=pd.DataFrame(columns=["cls","id"])
for i in range(len(labels)):
    #print("Processing label  :",i)
    tmpGroupBy=groupDataFrame.get_group(i)
    createdShuffleLabels=np.random.permutation(np.array(range(maxNum)))%labels[i]
    #print("Num of the label is : ",labels[i])
    lst=lst.append(tmpGroupBy.iloc[createdShuffleLabels],ignore_index=True)
    #print("Done")


# In[ ]:


lst = lst.sample(frac=1, random_state=24)


# In[ ]:


num_classes = 40
seed_everything(24)
lr          = 1e-3
IMG_SIZE    = 256


# In[ ]:


class MyDataset(Dataset):
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df['id'].iloc[idx]
        image = cv2.imread(os.path.join(train, img_name)+'.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        #image = Image.open(os.path.join(train, img_name))
        #image = image.convert("RGB")
        label = self.df['cls'].iloc[idx]
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


# In[ ]:


train_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

trainset     = MyDataset(lst, transform =train_transform)
train_loader = torch.utils.data.DataLoader(lst, batch_size=48, shuffle=True, num_workers=4)
valset       = MyDataset(val_df, transform = train_transform)
val_loader   = torch.utils.data.DataLoader(valset, batch_size=48, shuffle=False, num_workers=4)


# In[ ]:


model = EfficientNet.from_name('efficientnet-b4')
model.load_state_dict(torch.load('../input/efficientnet-pytorch/efficientnet-b4-e116e8b3.pth'))
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, 40)
model.cuda();


#refinelabelmodel = EfficientNet.from_name('efficientnet-b3')
#in_features = refinelabelmodel._fc.in_features
#refinelabelmodel._fc = nn.Linear(in_features, 40)
#refinelabelmodel.load_state_dict(torch.load('../input/garbage9291/acc_weight_best.pt'))


# In[ ]:


import torch
import torch.nn as nn


class LSR(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super().__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.reduction = reduction
    
    def _one_hot(self, labels, classes, value=1):
        """
            Convert labels to one hot vectors
        
        Args:
            labels: torch tensor in format [label1, label2, label3, ...]
            classes: int, number of classes
            value: label value in one hot vector, default to 1
        
        Returns:
            return one hot format labels in shape [batchsize, classes]
        """

        one_hot = torch.zeros(labels.size(0), classes)

        #labels and value_added  size must match
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)

        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)

        one_hot.scatter_add_(1, labels.long(), value_added)

        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        """convert targets to one-hot format, and smooth
        them.
        Args:
            target: target in form with [label1, label2, label_batchsize]
            length: length of one-hot format(number of classes)
            smooth_factor: smooth factor for label smooth
        
        Returns:
            smoothed labels in one hot format
        """
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        one_hot += smooth_factor / length

        return one_hot.to(target.device)

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                    .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                    .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                    .format(x.size()))


        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)

        if self.reduction == 'none':
            return loss
        
        elif self.reduction == 'sum':
            return torch.sum(loss)
        
        elif self.reduction == 'mean':
            return torch.mean(loss)
        
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


# In[ ]:


def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias, 
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture
    
    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)
        
        else: 
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)
        
    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]


# In[ ]:


import math
import torch
from torch.optim.optimizer import Optimizer, required
import itertools as it
#from torch.optim import Optimizer
#credit - Lookahead implementation from LonePatient - https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
#credit2 - RAdam code by https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py


class Ranger(Optimizer):
    
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, betas=(.9,0.999), eps=1e-8, weight_decay=0):
        #parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')
        
        #prep defaults and init torch.optim base
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params,defaults)
        
        #now we can get to work...
        for group in self.param_groups:
            group["step_counter"] = 0
            #print("group step counter init")
                      
        #look ahead params
        self.alpha = alpha
        self.k = k 
        
        #radam buffer for state
        self.radam_buffer = [[None,None,None] for ind in range(10)]
        
        #lookahead weights
        self.slow_weights = [[p.clone().detach() for p in group['params']]
                                for group in self.param_groups]
        
        #don't use grad for lookahead weights
        for w in it.chain(*self.slow_weights):
            w.requires_grad = False
        
    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)
       
        
    def step(self, closure=None):
        loss = None
        #note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.  
        #Uncomment if you need to use the actual closure...
        
        #if closure is not None:
            #loss = closure()
            
        #------------ radam
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
                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size
    
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
    
                if N_sma > 5:                    
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)
    
                p.data.copy_(p_data_fp32)
        
        
        #---------------- end radam step
        
        #look ahead tracking and updating if latest batch = k
        for group,slow_weights in zip(self.param_groups,self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p,q in zip(group['params'],slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.alpha,p.data - q.data)
                p.data.copy_(q.data)
            
        
            
        return loss


# In[ ]:


parameter = split_weights(model)
#optimizer = torch.optim.SGD(parameter, lr = 2e-2, momentum=0.9, weight_decay=1e-4, nesterov=True)
#optimizer = torch.optim.Adam(parameter, lr=lr, weight_decay=1e-5)
optimizer = Ranger(parameter, lr=1e-3, weight_decay=5e-4)
criterion =  LSR()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 3, 5, 7, 9, 20, 25, 30], gamma=0.5)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3, 4, 5,7, 8, 9, 10], gamma=0.5)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2, gamma=0.5)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)


# In[ ]:


def train_model(epoch):
    model.train() 
    avg_loss = 0.
    correct = 0
    total = 0
    for idx, (imgs, labels) in enumerate(train_loader):
        imgs_train, labels_train = imgs.cuda(), labels.cuda()
        output_train = model(imgs_train)
        loss = criterion(output_train,labels_train)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        #loss.backward()
        avg_loss += loss.item()
        optimizer.step() 
        optimizer.zero_grad()
        prediction = torch.argmax(output_train, 1)
        correct += (prediction == labels_train).sum().float()
        total += len(labels_train)
    acc = (correct/total).cpu().detach().data.numpy()
    avg_loss = avg_loss / (len(train_loader))
    return acc, avg_loss

def test_model():
    correct = 0
    avg_val_loss = 0.
    total = 0
    model.eval()
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(val_loader):
            imgs_vaild, labels_vaild = imgs.cuda(), labels.cuda()
            output_test = model(imgs_vaild)
            val_loss = criterion(output_test, labels_vaild)
            avg_val_loss += val_loss.item() 
            prediction = torch.argmax(output_test, 1)
            correct += (prediction == labels_vaild).sum().float()
            total += len(labels_vaild)
        val_acc = (correct/total).cpu().detach().data.numpy()
        avg_val_loss = avg_val_loss/ (len(val_loader)) 
    return val_acc, avg_val_loss


# In[ ]:


best_avg_loss = 100.0
best_avg_acc  = 0
n_epochs      = 50

for epoch in range(n_epochs):
    start_time   = time.time()
    acc, avg_loss = train_model(epoch)
    val_acc, avg_val_loss = test_model()
    elapsed_time = time.time() - start_time 
    print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t acc={:.4f} \t val_acc={:.4f} \t time={:.2f}s'.format(
        epoch + 1, n_epochs, avg_loss, avg_val_loss, acc, val_acc, elapsed_time))
    
    if avg_val_loss < best_avg_loss:
        best_avg_loss = avg_val_loss
        torch.save(model.state_dict(), 'loss_weight_best.pt')
    if val_acc>=best_avg_acc:
        best_avg_acc = val_acc
        torch.save(model.state_dict(), 'acc_weight_best.pt')
        
    scheduler.step()


# In[ ]:


get_ipython().system('rm -r ./apex')

