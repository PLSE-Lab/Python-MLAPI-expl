#!/usr/bin/env python
# coding: utf-8

# <center><img src="https://i.imgur.com/F4T8Ys2.jpg" width="600px"></center>

# # Introduction
# 
# Hello everyone! Welcome to the <font color="#1373f0">"ALASKA2 Image Steganalysis"</font> competition on Kaggle! In this competition, contestants are challenged to build machine learning models to predict whether a message has been <font color="#1373f0">hidden in an image using steganography</font>. An accurate solution to this problem can open up new possibilities in the area of steganalysis and deep learning-based encryption.
# 
# In this kernel, I will demonstrate how one can <font color="#1373f0">finetune EfficientNet-B3</font> to solve this task using PyTorch. I will use PyTorch v1.5 and Kaggle's <font color="#1373f0">TPU v3-8</font> to train the model on 8 folds and 1 epoch in less than three hours.

# # Acknowledgements
# 
# 1. [<font color="#1373f0">PyTorch XLA</font><font color="#5e5d5d"> ~ <u>by PyTorch</u></font>](https://pytorch.org/xla/release/1.5/index.html)
# 2. [<font color="#1373f0">Torchvision Models</font><font color="#5e5d5d"> ~ <u>by PyTorch</u></font>](https://pytorch.org/docs/stable/torchvision/models.html)
# 3. [<font color="#1373f0">PANDA / submit test</font><font color="#5e5d5d"> ~ <u>Yasufumi Nakama</u></font>](https://www.kaggle.com/yasufuminakama/panda-submit-test)
# 4. [<font color="#1373f0">Super-duper fast pytorch tpu kernel...</font><font color="#5e5d5d"> ~ <u>by Abhishek</u>](https://www.kaggle.com/abhishek/super-duper-fast-pytorch-tpu-kernel)

# # Contents
# 
# * [<font size=4 color="#1373f0">Preparing the ground</font>](#1)
#     * [<font color="#5e5d5d"><u>Set up PyTorch-XLA</u></font>](#1.1)
#     * [<font color="#5e5d5d"><u>Import libraries</u></font>](#1.2)
#     * [<font color="#5e5d5d"><u>Set hyperparameters and paths</u></font>](#1.3)
#     * [<font color="#5e5d5d"><u>Load .csv data</u></font>](#1.4)
#     * [<font color="#5e5d5d"><u>Display few images</u></font>](#1.5)
# 
#     
# * [<font size=4 color="#1373f0">Modeling</font>](#2)
#     * [<font color="#5e5d5d"><u>Build PyTorch dataset</u></font>](#2.1)
#     * [<font color="#5e5d5d"><u>Build EfficientNet-B3 model</u></font>](#2.2)
#     * [<font color="#5e5d5d"><u>Split 300, 000 images into 8 folds</u></font>](#2.3)
#     * [<font color="#5e5d5d"><u>Define cross entropy and accuracy</u></font>](#2.4)
#     * [<font color="#5e5d5d"><u>Define helper function for training logs</u></font>](#2.5)
#     * [<font color="#5e5d5d"><u>Train model on all 8 TPU cores in parallel</u></font>](#2.6)
# 
# 
# * [<font size=4 color="#1373f0">Takeaways</font>](#3)

# # Preparing the ground <a id="1"></a>

# ## Set up PyTorch-XLA <a id="1.1"></a> <font color="#1373f0" size=4>(inspired by Abhishek's kernel :D)</font>

# In[ ]:


get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev')
get_ipython().system('export XLA_USE_BF16=1')


# ## Import libraries <a id="1.2"></a> <font color="#1373f0" size=4>(for data loading, processing, and modeling on TPU)</font>

# In[ ]:


import os
import gc
import cv2
import sys
import time
import copy

import numpy as np
import pandas as pd
from colorama import Fore
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from keras.utils import to_categorical
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.utils.data
from torch.hub import load
from torch.optim import Adam
from torch import DoubleTensor, FloatTensor, LongTensor

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from torchvision.transforms import Normalize
from torch.utils.data import Dataset, DataLoader, sampler
from albumentations import VerticalFlip, HorizontalFlip, Compose


# ## Set hyperparamerters and paths <a id="1.3"></a> <font color="#1373f0" size=4>(adjust these to improve CV and LB :D)</font>

# In[ ]:


H = 512
W = 512
VF = 0.5
HF = 0.5
DELAY = 30
FRAC = 0.1
DROP = 0.225

FOLDS = 8
EPOCHS = 3
LR = 1e-3, 1e-3
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
MODEL_NAME = 'efficientnet_b3'
MODEL = 'rwightman/gen-efficientnet-pytorch'


# In[ ]:


PATH = '../input/'
MODEL_PATH = 'efficientnet_model'
DATA_PATH = PATH + 'alaska2-image-steganalysis/'
SAMPLE_SUB_PATH = DATA_PATH + 'sample_submission.csv'

TEST_PATH = DATA_PATH + 'Test/'
UERD_PATH = DATA_PATH + 'UERD/'
COVER_PATH = DATA_PATH + 'Cover/'
JMiPOD_PATH = DATA_PATH + 'JMiPOD/'
JUNIWARD_PATH = DATA_PATH + 'JUNIWARD/'
TRAIN_PATHS = [COVER_PATH, JMiPOD_PATH, JUNIWARD_PATH, UERD_PATH]


# ## Load .csv data <a id="1.4"></a> <font color="#1373f0" size=4>(to access image IDs for training and validation)</font>

# In[ ]:


sample_submission = pd.read_csv(SAMPLE_SUB_PATH)


# In[ ]:


sample_submission.head()


# ## Display few images <a id="1.5"></a> <font color="#1373f0" size=4>(from <i>Test</i> directory)</font>

# In[ ]:


def display_images(num):
    sq_num = np.sqrt(num)
    assert sq_num == int(sq_num)

    sq_num = int(sq_num)
    image_ids = os.listdir(TEST_PATH)
    fig, ax = plt.subplots(nrows=sq_num, ncols=sq_num, figsize=(20, 20))

    for i in range(sq_num):
        for j in range(sq_num):
            idx = i*sq_num + j
            img = cv2.imread(TEST_PATH + image_ids[idx])
            ax[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax[i, j].set_title('Image {}'.format(idx), fontsize=12)

    plt.show()


# In[ ]:


display_images(36)


# # Modeling <a id="2"></a>

# ## Build PyTorch dataset <a id="2.1"></a> <font color="#1373f0" size=4>(with image transforms and targets)</font>

# In[ ]:


def get_img(path, aug):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
    return aug(image=cv2.resize(img, (H, W)))['image']

class ALASKADataset(Dataset):
    def __init__(self, image_id, is_test, is_val):
        self.is_test = is_test
        self.image_id = image_id
        self.no_aug = is_test or is_val

        self.vertical = VerticalFlip(p=VF)
        self.horizontal = HorizontalFlip(p=HF)
        if self.no_aug: self.transform = lambda image: {'image': image}
        else: self.transform = Compose([self.vertical, self.horizontal], p=1)

    def __len__(self):
        multiplier = 1 if self.is_test else 4
        return multiplier*len(self.image_id)
    
    def __getitem__(self, idx):
        index = idx%len(self.image_id)

        if self.is_test:
            category = None
            path = TEST_PATH + self.image_id[index]
            return FloatTensor(get_img(path, self.transform))
        else:
            target = idx/len(self.image_id)
            category = [int(np.floor(target) > 0)]
            path = TRAIN_PATHS[int(target)] + self.image_id[index]
            return FloatTensor(get_img(path, self.transform)), FloatTensor(category)


# ## Build EfficientNet-B3 model <a id="2.2"></a> <font color="#1373f0" size=4>(with a custom Dense head)</font>

# In[ ]:


class ENSModel(nn.Module):
    def __init__(self):
        super(ENSModel, self).__init__()
        self.dropout = nn.Dropout(p=DROP)
        self.dense_output = nn.Linear(1536, 1)
        self.efn = load(MODEL, MODEL_NAME, pretrained=True)
        self.efn = nn.Sequential(*list(self.efn.children())[:-1])
        
    def forward(self, x):
        x = x.reshape(-1, 3, H, W)
        return self.dense_output(self.dropout(self.efn(x).reshape(-1, 1536)))


# ## Split 300,000 images into 8 folds <a id="2.3"></a> <font color="#1373f0" size=4>(for cross-validation)</font>

# In[ ]:


kfolds = KFold(n_splits=FOLDS)
image_id = os.listdir(COVER_PATH)
split_indices = kfolds.split(image_id)

val_ids, train_ids = [], []
for index in split_indices:
    val_ids.append(np.array(image_id)[index[1]])
    train_ids.append(np.array(image_id)[index[0]])


# ## Define binary cross entropy and accuracy <a id="2.4"></a> <font color="#1373f0" size=4>(for backpropagation)</font>

# In[ ]:


def bce(inp, targ):
    return nn.BCEWithLogitsLoss()(nn.Sigmoid()(inp), targ)

def acc(inp, targ):
    targ_idx = targ.squeeze()
    inp_idx = torch.round(nn.Sigmoid()(inp)).squeeze()
    return (inp_idx == targ_idx).float().sum(axis=0)/len(inp_idx)


# ## Define helper function for training logs <a id="2.5"></a> <font color="#1373f0" size=4>(to check training status)</font>

# In[ ]:


def print_metric(data, fold, start, end, metric, typ):
    n, value = "Steganalysis", np.round(data.item(), 3)
    g, c, y, r = Fore.GREEN, Fore.CYAN, Fore.YELLOW, Fore.RESET
    
    tick = g + '\u2714' + r
    t = typ, n, metric, c, value, r
    time = np.round(end - start, 1)
    time = "Time: {}{}{} s".format(y, time, r)
    string = "FOLD {} ".format(fold + 1) + tick + "  "
    print(string + "{} {} {}: {}{}{}".format(*t) + "  " + time)


# ## Train model on all 8 TPU cores in parallel <a id="2.6"></a> <font color="#1373f0" size=4>(one fold per core)</font>

# In[ ]:


class ImbSamp(sampler.Sampler):

    def __len__(self): return self.num_samples
    def __iter__(self): return (self.indices[i] for i in self._get_probs())
    def _get_label(self, dataset, idx): return int(idx/(len(self.dataset)/4) >= 1)
    
    def _get_weight(self, idx, count_dict):
        return 1.0/count_dict[self._get_label(self.dataset, idx)]
    
    def _get_probs(self):
        return torch.multinomial(self.weights, self.num_samples, replacement=True)

    def __init__(self, dataset, indices=None, num_samples=None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        count = {}
        self.dataset = dataset
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in count: count[label] += 1
            if label not in count: count[label] = 1

        self.weights = DoubleTensor([self._get_weight(idx, count) for idx in self.indices])


# In[ ]:


model = ENSModel()

def run(fold):
    val = val_ids[fold]
    train = train_ids[fold]
    device = xm.xla_device(fold + 1)

    val_set = ALASKADataset(val, False, True)
    train_set = ALASKADataset(train, False, False)
    val_loader = DataLoader(val_set, batch_size=VAL_BATCH_SIZE)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=ImbSamp(train_set))

    network = copy.deepcopy(model).to(device)
    optimizer = Adam([{'params': network.efn.parameters(), 'lr': LR[0]},
                      {'params': network.dense_output.parameters(), 'lr': LR[1]}])

    start = time.time()
    for epoch in range(EPOCHS):

        batch = 1
        for train_batch in train_loader:
            train_img, train_targs = train_batch
            
            network = network.to(device)
            train_img = train_img.to(device)
            train_targs = train_targs.to(device)
            
            network.train()
            train_preds = network.forward(train_img)
            train_loss = bce(train_preds, train_targs)

            optimizer.zero_grad()
            train_loss.backward()
            xm.optimizer_step(optimizer, barrier=True)

            batch = batch + 1
            if batch >= FRAC*len(train_loader): break

    network.eval()
    val_loss, val_acc = 0, 0
    for val_batch in tqdm(val_loader):

        img, targ = val_batch
        with torch.no_grad():
            img = img.to(device)
            targ = targ.to(device)
            network = network.to(device)
                
            pred = network.forward(img)
            val_acc += acc(pred, targ.squeeze(dim=1)).item()*len(pred)
            val_loss += bce(pred, targ.squeeze(dim=1)).item()*len(pred)

    end = time.time()
    time.sleep(DELAY*fold)
    network = network.cpu()
    model_path = MODEL_PATH + "_{}.pt"
    
    val_acc /= len(val_set)
    val_loss /= len(val_set)
    print_metric(val_acc, fold, start, end, metric="Accuracy", typ="Val")
    torch.save(network.state_dict(), model_path.format(fold + 1)); del network; gc.collect()


# In[ ]:


Parallel(n_jobs=FOLDS, backend="threading")(delayed(run)(i) for i in range(FOLDS))


# # Takeaways <a id="3"></a>
# 
# 1. Using all 8 TPU cores in parallel can dramatically speed up KFold training.
# 2. Using complex models (like ResNet-152, DenseNet-201, Efficient-B3, etc) can improve the model's performance.
# 3. Training and inference should be done in separate notebooks to avoid confusion and make it easier to iterate fast.
