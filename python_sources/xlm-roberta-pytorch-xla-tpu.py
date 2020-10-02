#!/usr/bin/env python
# coding: utf-8

# # Training kernel for XLM-RoBERTa using PyTorch on the TPU
# 
# ## If you found this helpful, please give it an upvote!
# 
# Using the amazing [PyTorch XLA library](https://github.com/pytorch/xla)
# 
# Validation will be added soon.
# 
# - V1: Proof-of-concept. It runs!
# - V6: Add validation code
# 
# Further work:
# 1. Tune LR further
# 2. xhlulu dataset (prelim. code works)
# 3. Further memory optimization tricks

# # Introduction
# 
# In this kernel, I will demonstrate how to properly train a larger model with [PyTorch XLA](https://pytorch.org/xla). PyTorch XLA allows one to train PyTorch models on [Google's tensor processing units (TPUs)](https://cloud.google.com/tpu). Kaggle provides 30 hours of free TPU compute.
# 
# Abhishek has already made a great PyTorch XLA kernel [here](https://www.kaggle.com/abhishek/bert-multi-lingual-tpu-training-8-cores-w-valid) for training a multi-lingual BERT model on a modified dataset (training on a sample of 20,000 rows from the data, plus validation set). However, xhlulu shared a [kernel](https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta) training an XLM-RoBERTa model using TPU and TensorFlow. xhlulu's uses a modified dataset, using the full train dataset downsampled.
# 
# This kernel is an attempt to train the XLM-RoBERTa model with PyTorch XLA. I use Abhishek's dataset to start out. I will attempt to use xhlulu's dataset in a follow-up kernel.
# 
# 
# There are a couple challenges that make PyTorch XLA a little harder to use:
# 1. The API is much more lower-level compared to TensorFlow. PyTorch XLA requires you to use XLA-specific dataloaders and XLA-specific optimizer stepping. Additionally, it requires the definition of a train/evaluation loop function that needs to be spawned using PyTorch XLA's multiprocessing functionality. PyTorch Lightning's [TPU support](https://pytorch-lightning.readthedocs.io/en/latest/tpu.html) may help alleviate some of these issues, but code may still need to be optimized further, at least when using Kaggle TPUs, due to the low amount of RAM available. TensorFlow simple requires the definition of a TPU distribution strategy, and you are set to go.
# 2. PyTorch XLA works differently compared to TensorFlow, leading to higher host VM memory usage with PyTorch XLA compared to TensorFlow. Specifically, PyTorch XLA builds the XLA graphs, initializes the weights, runs input pipelines etc and then feeds them to the TPUs. On the other hand, TensorFlow hands the XLA graphs to the TPU and the TPU does most of the heavy lifting. This difference is amplified by the fact that the appropriate data and model needs to be replicated 8 times to utilize all eight cores of the TPU.
# 
# In this kernel, we will see some additional optimizations that will allow us to train the XLM-R model. There are _even more_ optimizations that could be done, which will be described in a future kernel.

# # Imports
# 
# The code cell below will install PyTorch XLA.

# In[ ]:


get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev')


# Here are all of our imports. You will note that there are already an optimization applied in order for PyTorch XLA to train.
# 
# `XLA_USE_BF16` is an environment variable that tells PyTorch XLA to automatically use [bfloat16](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus).

# In[ ]:


import os
os.environ['XLA_USE_BF16']="1"
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'

import torch
import pandas as pd
from scipy import stats
import numpy as np

from tqdm import tqdm
from collections import OrderedDict, namedtuple
import torch.nn as nn
from torch.optim import lr_scheduler
import joblib

import logging
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
import sys
from sklearn import metrics, model_selection


# Here we import all the PyTorch XLA-specific modules.

# In[ ]:


import warnings
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


class AverageMeter:
    """
    Computes and stores the average and current value
    """
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


# # Dataset and Model

# Here is a simple class to create datasets from numpy arrays. I think this dataset class could likely be further optimized for decreased memory usage.

# In[ ]:


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self,*arrays):
        assert all(arrays[0].shape[0] == array.shape[0] for array in arrays)
        self.arrays = arrays
    
    def __getitem__(self, index):
        return tuple(torch.from_numpy(np.array(array[index])) for array in self.arrays)
    
    def __len__(self):
        return self.arrays[0].shape[0]


# Here is the XLM-RoBERTa model definition inspired by the model used in xhlulu's kernel.

# In[ ]:


class CustomRoberta(nn.Module):
    def __init__(self):
        super(CustomRoberta, self).__init__()
        self.num_labels = 1
        self.roberta = transformers.XLMRobertaModel.from_pretrained("xlm-roberta-large", output_hidden_states=False, num_labels=1)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(1024, self.num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None):

        _, o2 = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)

        logits = self.classifier(o2)       
        outputs = logits
        return outputs


# Let's define our model. If you check the PyTorch XLA documentation, you will note that the recommended setup is to define the model in the function that is run on each of the 8 cores. But doing so will lead to high VM memory usage. Therefore, this is the setup used for low-memory VMs.

# In[ ]:


mx = CustomRoberta();
mx


# The trick also used in xhlulu's kernel is to pre-tokenize the dataset. This is done [over here](https://www.kaggle.com/tanlikesmath/xlm-r-large-tokenize-dataset). It uses the same `regular_encode` function defined in xhlulu's kernel. We now load it over here. 
# 
# An additional trick used that may help memory usage is to load the dataset as a memory-mapped dataset (`mmap_mode='r'`). This way, the whole dataset isn't in the RAM at the same time.

# In[ ]:


tokenized_path = '../input/xlm-r-large-tokenize-dataset/'


# In[ ]:


x_train = np.load(tokenized_path+'x_train.npy',mmap_mode='r')
train_toxic = np.load(tokenized_path+'df_train_toxic.npy',mmap_mode='r')

x_valid = np.load(tokenized_path+'x_valid.npy',mmap_mode='r')
valid_toxic = np.load(tokenized_path+'df_valid_toxic.npy',mmap_mode='r')


# In[ ]:


x_train.shape, x_valid.shape


# Let's create our dataset!

# In[ ]:


train_dataset = ArrayDataset(x_train, train_toxic)
valid_dataset = ArrayDataset(x_valid, valid_toxic)


# Make sure to delete any unused variables:

# In[ ]:


del x_train, x_valid
import gc;gc.collect()


# In[ ]:


gc.collect()


# # Training

# In[ ]:


import torch_xla.version as xv
print('PYTORCH:', xv.__torch_gitrev__)
print('XLA:', xv.__xla_gitrev__)


# In[ ]:


get_ipython().system('free -h')


# In[ ]:


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


# In[ ]:


def reduce_fn(vals):
    return sum(vals) / len(vals)


# We now define our training loop and evaluation loop functions.
# 
# To get the loss of a batch, since the data is spread across the 8 cores, we have to _reduce_ the loss.
# 
# PyTorch XLA requires that the optimizer be stepped using their own function `xm.optimizer_step(optimizer)`.

# In[ ]:


def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    for bi, d in enumerate(data_loader):

        ids = d[0]
        targets = d[1]

        ids = ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(
            input_ids=ids,
        )
        loss = loss_fn(outputs, targets)
        if bi % 50 == 0:
            loss_reduced = xm.mesh_reduce('loss_reduce',loss,reduce_fn)
            xm.master_print(f'bi={bi}, loss={loss_reduced}')
        loss.backward()
        xm.optimizer_step(optimizer)
        if scheduler is not None:
            scheduler.step()
            

    model.eval()
    
def eval_loop_fn(data_loader, model, device):
    fin_targets = []
    fin_outputs = []
    for bi, d in enumerate(data_loader):
        ids = d[0]
        targets = d[1]

        ids = ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(
            input_ids=ids,
        )

        targets_np = targets.cpu().detach().numpy().tolist()
        outputs_np = outputs.cpu().detach().numpy().tolist()
        fin_targets.extend(targets_np)
        fin_outputs.extend(outputs_np)    
        del targets_np, outputs_np
        gc.collect()
    return fin_outputs, fin_targets


# We finally define our function that will be spawned by PyTorch XLA multiprocessing. This function will be run on each of the 8 cores. There are several things to note:
# 1. We need to use a `DistributedSampler` that will appropriately distribute the dataset across the 8 cores.
# 2. We are using `num_workers=0` as that decreases memory usage (only master process loading data).
# 3. We put the model onto the TPU
# 4. We use `ParallelLoader` which is a PyTorch XLA-specific DataLoader for loading data onto the TPU.

# In[ ]:


def _run():
    MAX_LEN = 192
    TRAIN_BATCH_SIZE = 16
    EPOCHS = 2

    train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        drop_last=True,
        num_workers=0,
    )
    
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
          valid_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=4,
        sampler=valid_sampler,
        drop_last=False,
        num_workers=0
    )

    device = xm.xla_device()
    model = mx.to(device)
    xm.master_print('done loading model')

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    lr = 0.5e-5 * xm.xrt_world_size()
    num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE / xm.xrt_world_size() * EPOCHS)
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')


    for epoch in range(EPOCHS):
        gc.collect()
        para_loader = pl.ParallelLoader(train_data_loader, [device])
        xm.master_print('parallel loader created... training now')
        gc.collect()
        train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler)
        del para_loader
        para_loader = pl.ParallelLoader(valid_data_loader, [device])
        gc.collect()
        o, t = eval_loop_fn(para_loader.per_device_loader(device), model, device)
        del para_loader
        gc.collect()
        auc = metrics.roc_auc_score(np.array(t) >= 0.5, o)
        auc_reduced = xm.mesh_reduce('auc_reduce',auc,reduce_fn)
        xm.master_print(f'AUC = {auc_reduced}')
        gc.collect()
    xm.save(model.state_dict(), "xlm_roberta_model.bin")


# # Start training!
# 
# Let's spawn the `_mp_fn` and start training!

# In[ ]:


import time

# Start training processes
def _mp_fn(rank, flags):
    a = _run()

FLAGS={}
start_time = time.time()
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')


# In[ ]:


print('Time taken: ',time.time()-start_time)


# # Acknowledgments:
# - Based on data from [Abhishek's code](https://www.kaggle.com/abhishek/bert-multi-lingual-tpu-training-8-cores-w-valid)
# - Model based on [xhlulu's code](https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta)
# - Original attempt from [Aditya's code](https://www.kaggle.com/adityaecdrid/simple-xlmr-tpu-pytorch)
# - Discussion with Davide Libenzi and Daniel Sohn (PyTorch XLA team) - [code](https://www.kaggle.com/davidelibenzi/simple-xlmr-tpu-pytorch)
# - Fruitful discussions with Abhishek and Aditya
# 
# 
# # Fin
# 
# If you have any questions or suggestions, please drop a comment! :)
