#!/usr/bin/env python
# coding: utf-8

# ## Using all 8 cores of a v3 TPU using pytorch/XLA
# Folked from https://www.kaggle.com/dhananjay3/pytorch-xla-for-tpu-with-multiprocessing
# I added some codes to create the submission.csv.

# ## setup for pytorch/xla on TPU
# thanks to this [kernel](https://www.kaggle.com/byrachonok/pytorch-xla-for-tpuv3)

# In[ ]:


import os
import collections
from datetime import datetime, timedelta

os.environ["XRT_TPU_CONFIG"] = "tpu_worker;0;10.0.0.2:8470"

_VersionConfig = collections.namedtuple('_VersionConfig', 'wheels,server')
VERSION = "torch_xla==nightly"
CONFIG = {
    'torch_xla==nightly': _VersionConfig('nightly', 'XRT-dev{}'.format(
        (datetime.today() - timedelta(1)).strftime('%Y%m%d')))}[VERSION]

DIST_BUCKET = 'gs://tpu-pytorch/wheels'
TORCH_WHEEL = 'torch-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)
TORCH_XLA_WHEEL = 'torch_xla-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)
TORCHVISION_WHEEL = 'torchvision-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)

get_ipython().system('export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH')
get_ipython().system('apt-get install libomp5 -y')
get_ipython().system('apt-get install libopenblas-dev -y')

get_ipython().system('pip uninstall -y torch torchvision')
get_ipython().system('gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" .')
get_ipython().system('gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" .')
get_ipython().system('gsutil cp "$DIST_BUCKET/$TORCHVISION_WHEEL" .')
get_ipython().system('pip install "$TORCH_WHEEL"')
get_ipython().system('pip install "$TORCH_XLA_WHEEL"')
get_ipython().system('pip install "$TORCHVISION_WHEEL"')


# In[ ]:





# ## Imports

# In[ ]:


import os
import re
import cv2
import time
import tensorflow
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
import requests, threading
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

import torch_xla
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
torch.set_default_tensor_type('torch.FloatTensor')


# In[ ]:


os.listdir('../input')


# In[ ]:


test_jpeg = glob('../input/104-flowers-garden-of-eden/jpeg-224x224/test/*')
len(test_jpeg)


# In[ ]:


# do not uncomment see https://github.com/pytorch/xla/issues/1587

# xm.get_xla_supported_devices()
# xm.xrt_world_size() # 1


# ## Dataset

# In[ ]:


# DATASET_DIR = '/kaggle/input/104-flowers-garden-of-eden/jpeg-512x512'
DATASET_DIR = '/kaggle/input/104-flowers-garden-of-eden/jpeg-224x224'
TRAIN_DIR  = DATASET_DIR + '/train'
VAL_DIR  = DATASET_DIR + '/val'
TEST_DIR  = DATASET_DIR + '/test'
BATCH_SIZE = 16 # per core 
NUM_EPOCH = 25#25


# In[ ]:


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.ToTensor(),
                                      normalize])

valid_transform = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      normalize])


# In[ ]:


train_df=pd.read_csv('../input/create-pytorch-xla-datas/train.csv')
train_df.head()


# In[ ]:


valid_df=pd.read_csv('../input/create-pytorch-xla-datas/valid.csv')
valid_df.head()


# In[ ]:


sub=pd.read_csv('../input/getting-started-with-100-flowers-on-tpu/submission.csv')
sub.head()


# In[ ]:


test_img_ids=sub['id'].values


# In[ ]:


import cv2


# In[ ]:


class MyDataset(Dataset):
    def __init__(self, data,  transform=None):
        self.img_ids = data
        
        
        self.transform = transform
        
    def __getitem__(self, index):
#         print('index',index)
        img_id=self.img_ids[index]

        img=  Image.open(f'{TEST_DIR}/{img_id}.jpeg')
#         img = cv2.imread(f'{TEST_DIR}/{img_id}.jpeg')
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)

        return img
    
    def __len__(self):
        return len(self.img_ids)
    
    
class MyDFDataset(Dataset):
    def __init__(self, img_ids,labels,  transform):
        self.img_ids = img_ids
        self.labels=labels
        self.transform = transform
        
    def __getitem__(self, index):
#         print('index',index)
        img_id=self.img_ids[index]

        img=  Image.open(img_id).convert('RGB')
#         img = cv2.imread(f'{TEST_DIR}/{img_id}.jpeg')
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = self.transform(img)
        label=self.labels[index]
        return img,label
    
    def __len__(self):
        return len(self.img_ids)       


# In[ ]:


mydata=MyDataset(test_img_ids)
# mydata


# test_loader = torch.utils.data.DataLoader(
#         mydata,
#         batch_size=128,
#         shuffle=False,
#         num_workers=0,
#         drop_last=False)
# 
# for b in test_loader:
#     print(b.shape)
#     break

# In[ ]:


train=MyDFDataset(train_df['img_id'].values,train_df['label'].values,transform=train_transform)

valid=MyDFDataset(valid_df['img_id'].values,valid_df['label'].values,transform=valid_transform)


# In[ ]:


# train = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
# valid = datasets.ImageFolder(VAL_DIR, transform=valid_transform)
test = MyDataset(test_img_ids, transform=valid_transform)
# print out some data stats
print('Num training images: ', len(train))
print('Num valid images: ', len(valid))
print('Num test images: ', len(test))


# ## Model

# In[ ]:


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        
        self.base_model = torchvision.models.densenet201(pretrained=True)
        self.base_model.classifier = nn.Identity()
        self.fc = torch.nn.Sequential(
                    torch.nn.Linear(1920, 1024, bias = True),
                    torch.nn.BatchNorm1d(1024),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(1024, 512, bias = True),
                    torch.nn.BatchNorm1d(512),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(512, 104))
        
    def forward(self, inputs):
        x = self.base_model(inputs)
        return self.fc(x)


# In[ ]:


model = MyModel()
print(model)
del model


# ## Training

# In[ ]:


def train_model():
    global train, valid
    
    torch.manual_seed(42)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
        drop_last=True) # print(len(train_loader))
    
    valid_loader = torch.utils.data.DataLoader(
        valid,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True)
    
    xm.master_print(f"Train for {len(train_loader)} steps per epoch")
    # Scale learning rate to num cores
    learning_rate = 0.0001 * xm.xrt_world_size()

    # Get loss function, optimizer, and model
    device = xm.xla_device()

    model = MyModel()
    
    for param in model.base_model.parameters(): # freeze some layers
        param.requires_grad = False
    
    model = model.to(device)
    loss_fn =  nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = OneCycleLR(optimizer, 
                           learning_rate, 
                           div_factor=10.0, 
                           final_div_factor=50.0, 
                           epochs=NUM_EPOCH,
                           steps_per_epoch=len(train_loader))
    
    
    
    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        for x, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(data.shape[0])
            scheduler.step()
            if x % 30 == 0:
                print('[xla:{}]({})\tLoss={:.3f}\tRate={:.2f}\tGlobalRate={:.2f}'.format(
                    xm.get_ordinal(), x, loss.item(), tracker.rate(),
                    tracker.global_rate()), flush=True)

    def test_loop_fn(loader):
        with torch.no_grad():
            total_samples, correct = 0, 0
            model.eval()
            for data, target in loader:
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size()[0]
            accuracy = 100.0 * correct / total_samples
            print('[xla:{}] Accuracy={:.2f}%'.format(xm.get_ordinal(), accuracy), flush=True)
            model.train()
        return accuracy

    # Train and eval loops
    accuracy = []
    for epoch in range(1, NUM_EPOCH + 1):
        start = time.time()
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device))
        para_loader = pl.ParallelLoader(valid_loader, [device])
        accuracy.append(test_loop_fn(para_loader.per_device_loader(device)))
        xm.master_print("Finished training epoch {} acc {:.2f} in {:.2f} sec"                        .format(epoch, accuracy[-1], time.time() - start))        
        xm.save(model.state_dict(), "./model.pt")
        
#         if epoch == 15: #unfreeze
#                 for param in model.base_model.parameters():
#                     param.requires_grad = True

    return accuracy


# In[ ]:


# Start training processes
def _mp_fn(rank, flags):
    global acc_list
    torch.set_default_tensor_type('torch.FloatTensor')
    acc = train_model()

FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')


# ### Doing validation every epoch increases time per epoch significantly. faster version of this notebook is in comments.

# In[ ]:





# # predict

# In[ ]:


get_ipython().run_cell_magic('time', '', "print('start to predict...')\ndevice = xm.xla_device()\n\nmodel = MyModel()\nmodel.eval()\nwith torch.no_grad():\n    model = model.to(device)\n    model.load_state_dict(torch.load('./model.pt'))\n    test_loader = torch.utils.data.DataLoader(\n        test,\n        batch_size=BATCH_SIZE,\n        shuffle=False,\n        num_workers=0,\n        drop_last=False)\n    para_loader = pl.ParallelLoader(test_loader, [device])\n    preds=[]\n    for data in para_loader.per_device_loader(device):\n        output = model(data)\n        pred= torch.argmax(output, dim=1).int().cpu().numpy()\n        preds.extend(pred)\n    print(len(preds))\n    ")


# In[ ]:


sub['label']=preds
sub.to_csv('submission.csv',index=False)
sub.head()

