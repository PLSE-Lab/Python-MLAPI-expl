#!/usr/bin/env python
# coding: utf-8

# ## Code taken from [https://www.kaggle.com/abhishek/bert-multi-lingual-tpu-training-8-cores-w-valid](http://) and modified for this comeptetion. Thank you [Abhishek](http://www.kaggle.com/abhishek) :)

# # Prolouge:
# ### Welcome to my kaggle adventures of learning how to use a tpu in kaggle. Let's learn together.
# Hey there, This is a kernel to teach you how to train on a tpu in kaggle on all cores parellely. This a work in progress kernel as I will keep it updating as I learn new things!  

# ### Torch XLA setup

# In[ ]:


# for TPU
get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev')


# In[ ]:


get_ipython().system('pip install efficientnet_pytorch > /dev/null')
get_ipython().system('pip install albumentations > /dev/null')


# ### Imports required for TPU

# In[ ]:


import warnings
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import warnings

warnings.filterwarnings("ignore")


# In[ ]:


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import operator
from PIL import Image 
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Resize
from efficientnet_pytorch import EfficientNet
from transformers import AdamW, get_cosine_schedule_with_warmup
from albumentations import *
from albumentations.pytorch import ToTensor
from tqdm import tqdm
import json
import time


# In[ ]:


BASE_DIR = '../input/plant-pathology-2020-fgvc7/'


# In[ ]:


train_df = pd.read_csv(BASE_DIR +'train.csv')


# In[ ]:


train_df.head()


# ### make training labels by taking argmax
# 

# In[ ]:


train_df['image_id'] = BASE_DIR + 'images/' + train_df['image_id'] + '.jpg'


# In[ ]:


train_df['label'] = [np.argmax(label) for label in train_df[['healthy','multiple_diseases','rust','scab']].values]


# In[ ]:


train_df.head()


# # Simple PyTorch Training
# 
# 1. In this part let's try simple pytorch model and train it for 20 epochs straight without CV.
# 2. The model of my choice is: 'EfficientNet-b5'.
# 3. Parellely training on all 8 cores

# In[ ]:


class SimpleDataset(Dataset):
    def __init__(self, image_ids_df, labels_df, transform=None):
        self.image_ids = image_ids_df
        self.labels = labels_df
        self.transform = transform
        
    def __getitem__(self, idx):
        image = cv2.imread(self.image_ids.values[idx])
        label = self.labels.values[idx]
        
        sample = {
            'image': image,
            'label': label
        }
        
        if self.transform:
            sample = self.transform(**sample)
        
        image, label = sample['image'], sample['label']
        
        return image, label
    
    def __len__(self):
        return len(self.image_ids)
    

        


# In[ ]:


image_ids = train_df['image_id']
labels = train_df['label']


# ## Splitting the dataset:
# I split the datset simply using sklearn's train_test_split()

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(image_ids, labels, test_size=0.25, random_state=42)


# In[ ]:


train_transform = Compose(
    [
        Resize(224, 224),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
#         ShiftScaleRotate(rotate_limit=25.0, p=0.7),
#         OneOf(
#             [
#                 IAAEmboss(p=1),
#                 IAASharpen(p=1),
#                 Blur(p=1)
#             ], 
#             p=0.5
#         ),
#         IAAPiecewiseAffine(p=0.5),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), always_apply=True),
        ToTensor()
    ]
)


# In[ ]:


model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=4)


# # My observations:
# 1. torch.xla has it's own specific requirements. U can't simply make a device using `xm.xla_device()` and pass the model to it. 
# <br/><br/>
# With that:
#     1. Optimizer has to stepped with `xm.optimizer_step(optimizer)`.
#     2. You have to save the model with `xm.save(model.state_dict(), '<your-model-name>)`
#     3. You have to use `xm.master_print(...)` to print. This you can try for yourself below. Try to change 
#        the `xm.master_print(f'Batch: {batch_idx}, loss: {loss.item()}')` in the training function(`train_fn()`) to simple 
#        `print(f'Batch: {batch_idx}, loss: {loss.item()}')`. You will see it deos not get printed.
#     4. For parellel training we first define the distributed train & valid sampler, then we wrap the dataloaders in `torch_xla.distributed.parallel_loader(<your-data-loader>)` and create a `torch_xla.distributed.parallel_loader` object 
#     5. While passing it to training and validation function we specify this `para_loader.per_device_loader(device)`. This is what you will iterate over in the training function, i.e. we pass a parelleloader and not a dataloader (for parellel training only).

# In[ ]:


def _run(model):
     
    def train_fn(epoch, train_dataloader, optimizer, criterion, scheduler, device):

        running_loss = 0
        total = 0
        model.train()

        for batch_idx, (images, labels) in enumerate(train_dataloader, 1):

            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            xm.master_print(f'Batch: {batch_idx}, loss: {loss.item()}')

            loss.backward()
            xm.optimizer_step(optimizer)

            lr_scheduler.step()

    def valid_fn(epoch, valid_dataloader, criterion, device):

        running_loss = 0
        total = 0
        preds_acc = []
        labels_acc = []

        model.eval()

        for batch_idx, (images, labels) in enumerate(valid_dataloader, 1):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            
            xm.master_print(f'Batch: {batch_idx}, loss: {loss.item()}')

            running_loss += loss.item()
    
    
    EPOCHS = 20
    BATCH_SIZE = 64
    
    train_dataset = SimpleDataset(X_train, y_train, transform=train_transform)
    valid_dataset = SimpleDataset(X_test, y_test, transform=train_transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
          valid_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, sampler=valid_sampler, num_workers=1)
    
    device = xm.xla_device()
    model = model.to(device)
    
    lr = 0.4 * 1e-5 * xm.xrt_world_size()
    criterion = nn.CrossEntropyLoss()
    
    optimizer = AdamW(model.parameters(), lr=lr)
    num_train_steps = int(len(train_dataset) / BATCH_SIZE / xm.xrt_world_size() * EPOCHS)
    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')
    num_train_steps = int(len(train_dataset) / BATCH_SIZE * EPOCHS)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    
    train_loss = []
    valid_loss = []
    best_loss = 1
    
    train_begin = time.time()
    for epoch in range(EPOCHS):
        
        para_loader = pl.ParallelLoader(train_dataloader, [device])

        start = time.time()
        print('*'*15)
        print(f'EPOCH: {epoch+1}')
        print('*'*15)

        print('Training.....')
        train_fn(epoch=epoch+1, 
                                  train_dataloader=para_loader.per_device_loader(device), 
                                  optimizer=optimizer, 
                                  criterion=criterion,
                                  scheduler=lr_scheduler,
                                  device=device)


        
        with torch.no_grad():
            
            para_loader = pl.ParallelLoader(valid_dataloader, [device])
            
            print('Validating....')
            valid_fn(epoch=epoch+1, 
                                      valid_dataloader=para_loader.per_device_loader(device), 
                                      criterion=criterion, 
                                      device=device)
            xm.save(
                model.state_dict(),
                f'efficientnet-b0-bs-8.pt'
            )
    
        print(f'Epoch completed in {(time.time() - start)/60} minutes')
    print(f'Training completed in {(time.time() - train_begin)/60} minutes')


# ## Below cell is what makes the model train on all 8 cores! Run yourself to see the magic

# In[ ]:


# Start training processes
def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    a = _run(model)

FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')


# # Epilouge: 
# 
# 1. With parellely running on all 8 cores, my training time was 15 minutes (20 epochs) with a batch size of 64 for training and 32 for validation, as opposed to 1 hr on my local device (which has a gtx 1050 with 4 gb memory) with a batch size of 8 for both training and validation. Okay I get it, it's not a fair comparison as we have a more powerful gpu on kaggle, but I am guessing you get what I am trying to say. :P  
# 2. This kernel was for me to keep as a future reference, but I want to share it with all of you. You are the ones from whom I learn so much.
# 3. About the accuracy and inference that I haven't done and I am currently working on. I just ran it once to see how long it takes. :P
# 
# You're free to correct me, make suggestions and tell me on what I can improve on. :) 
# Lastly if u found it any useful please consider to upvote :)
# 
# Also for more information: [Torch XLA documentation](https://pytorch.org/xla/release/1.5/index.html#)

# In[ ]:




