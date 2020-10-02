#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will introduce a baseline using catalyst. I perfer the `console` version rather than this kind of `notebook` version. 
# However, I am going to try my best to make it to be more clear. 
# In our local experiment, I got 0.953 CV (one fold) and 0.954 LB (one fold) with following settings:
# * Resnet34 + 3 heads (refer the code bellow) 
# * Optimizer AdamW 
# * Loss: CrossEntropyLoss for each head
# * Strategies: 
#     * Freeze the backbone:
#         * lr = 0.001
#         * num_epochs = 3
#     * Unfreeze the backbone:
#         * lr = 0.0001
#         * num_epochs = 15
#         * Scheduler: OneCycleLRWithWarmup:
#             * num_steps: 15
#             * warmup_steps: 5
#             * lr_range: [0.0005, 0.00001]
#             * momentum_range: [0.85, 0.95]
#             
#             
# In this notebook, the settings are different from my exp above because of computing resource and time limitation. You are welcome to experiment in your local environment

# In[ ]:





# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install cnn-finetune')


# In[ ]:


from typing import Callable, List, Tuple

import os
import torch
import catalyst

from catalyst.dl import utils

print(f"torch: {torch.__version__}, catalyst: {catalyst.__version__}")

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "" - CPU, "0" - 1 GPU, "0,1" - MultiGPU

SEED = 2411
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)


# # Model

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
from cnn_finetune import make_model


class BegaliaiModel(nn.Module):
    def __init__(self, model_name, pretrained, num_classes):
        super(BegaliaiModel, self).__init__()
        self.model = make_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=1000,
        )

        in_features = self.model._classifier.in_features

        self.head_grapheme_root = nn.Linear(in_features, num_classes[0])
        self.head_vowel_diacritic = nn.Linear(in_features, num_classes[1])
        self.head_consonant_diacritic = nn.Linear(in_features, num_classes[2])

    def freeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.model._features(x)
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)

        logit_grapheme_root = self.head_grapheme_root(features)
        logit_vowel_diacritic = self.head_vowel_diacritic(features)
        logit_consonant_diacritic = self.head_consonant_diacritic(features)

        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic


# # Dataset

# In[ ]:


import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def load_image(path):
    image = cv2.imread(path, 0)
    image = np.stack((image, image, image), axis=-1)
    return image


class BengaliaiDataset(Dataset):

    def __init__(self, df, root, transform):
        self.image_ids = df['image_id'].values
        self.grapheme_roots = df['grapheme_root'].values
        self.vowel_diacritics = df['vowel_diacritic'].values
        self.consonant_diacritics = df['consonant_diacritic'].values

        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        grapheme_root = self.grapheme_roots[idx]
        vowel_diacritic = self.vowel_diacritics[idx]
        consonant_diacritic = self.consonant_diacritics[idx]

        image_id = os.path.join(self.root, image_id + '.png')
        image = load_image(image_id)
        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            'images': image,
            'grapheme_roots': grapheme_root,
            'vowel_diacritics': vowel_diacritic,
            'consonant_diacritics': consonant_diacritic
        }


# In[ ]:





# # Augmentations

# In[ ]:


from albumentations import Compose, Resize, Rotate, HorizontalFlip, Normalize

def train_aug(image_size):
    return Compose([
        Resize(*image_size),
        Rotate(10),
        HorizontalFlip(),
        Normalize()
    ], p=1)


def valid_aug(image_size):
    return Compose([
        Resize(*image_size),
        Normalize()
    ], p=1)


# # Callbacks

# This is a callback for `hierarchical macro-averaged recall`

# In[ ]:


from typing import Any, List, Optional, Union  # isort:skip
# import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from catalyst.dl.core import Callback, CallbackOrder, RunnerState
from sklearn.metrics import recall_score
import numpy as np

class HMacroAveragedRecall(Callback):
    def __init__(
        self,
        input_grapheme_root_key: str = "grapheme_roots",
        input_consonant_diacritic_key: str = "consonant_diacritics",
        input_vowel_diacritic_key: str = "vowel_diacritics",

        output_grapheme_root_key: str = "logit_grapheme_root",
        output_consonant_diacritic_key: str = "logit_consonant_diacritic",
        output_vowel_diacritic_key: str = "logit_vowel_diacritic",

        prefix: str = "hmar",
    ):
        self.input_grapheme_root_key = input_grapheme_root_key
        self.input_consonant_diacritic_key = input_consonant_diacritic_key
        self.input_vowel_diacritic_key = input_vowel_diacritic_key

        self.output_grapheme_root_key = output_grapheme_root_key
        self.output_consonant_diacritic_key = output_consonant_diacritic_key
        self.output_vowel_diacritic_key = output_vowel_diacritic_key
        self.prefix = prefix

        super().__init__(CallbackOrder.Metric)

    def on_batch_end(self, state: RunnerState):
        input_grapheme_root = state.input[self.input_grapheme_root_key].detach().cpu().numpy()
        input_consonant_diacritic = state.input[self.input_consonant_diacritic_key].detach().cpu().numpy()
        input_vowel_diacritic = state.input[self.input_vowel_diacritic_key].detach().cpu().numpy()

        output_grapheme_root = state.output[self.output_grapheme_root_key]
        output_grapheme_root = F.softmax(output_grapheme_root, 1)
        _, output_grapheme_root = torch.max(output_grapheme_root, 1)
        output_grapheme_root = output_grapheme_root.detach().cpu().numpy()

        output_consonant_diacritic = state.output[self.output_consonant_diacritic_key]
        output_consonant_diacritic = F.softmax(output_consonant_diacritic, 1)
        _, output_consonant_diacritic = torch.max(output_consonant_diacritic, 1)
        output_consonant_diacritic = output_consonant_diacritic.detach().cpu().numpy()

        output_vowel_diacritic = state.output[self.output_vowel_diacritic_key]
        output_vowel_diacritic = F.softmax(output_vowel_diacritic, 1)
        _, output_vowel_diacritic = torch.max(output_vowel_diacritic, 1)
        output_vowel_diacritic = output_vowel_diacritic.detach().cpu().numpy()


        scores = []
        scores.append(recall_score(input_grapheme_root, output_grapheme_root, average='macro'))
        scores.append(recall_score(input_consonant_diacritic, output_consonant_diacritic, average='macro'))
        scores.append(recall_score(input_vowel_diacritic, output_vowel_diacritic, average='macro'))

        final_score = np.average(scores, weights=[2, 1, 1])
        state.metrics.add_batch_value(name=self.prefix, value=final_score)


# # Loaders

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


import collections

data_root = "../input/bengaliai/256_train/256/"
df = pd.read_csv("../input/bengaliai-cv19/train.csv")
train_df, valid_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=2411)

# image size
image_size = [224, 224]

# transforms 
train_transform = train_aug(image_size)
valid_transform = valid_aug(image_size)

train_dataset = BengaliaiDataset(
    df=train_df, 
    root=data_root, 
    transform=train_transform
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    drop_last=False
)


valid_dataset = BengaliaiDataset(
    df=valid_df, 
    root=data_root, 
    transform=valid_transform
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=32,
    num_workers=4,
    shuffle=False,
    drop_last=False
)

loaders = collections.OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = valid_loader


# # Experiments

# ## Criterions

# You can train each head with different loss functions, just define it as a dictionary

# In[ ]:


from torch import nn

# we have multiple criterions
criterion = {
    "ce": nn.CrossEntropyLoss(),
    # Define your awesome losses in here. Ex: Focal, lovasz, etc
}


# In[ ]:





# ## Model

# In[ ]:


model = BegaliaiModel(
    model_name='resnet34',
    num_classes=[168, 11, 7],
    pretrained=True
)


# # Freeze the network

# In[ ]:


from torch import optim
from catalyst.contrib.optimizers import RAdam, Lookahead
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, IouCallback,   CriterionCallback, CriterionAggregatorCallback


# In[ ]:


model.freeze()


# In[ ]:


learning_rate = 0.001
optimizer = optim.AdamW(
    model.parameters(), 
    lr=learning_rate
)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.3) # Hack


# In[ ]:


num_epochs = 3
logdir = "./logs/bengaliai/"

device = utils.get_device()
print(f"device: {device}")

# by default SupervisedRunner uses "features" and "targets",
# in our case we get "image" and "mask" keys in dataset __getitem__
runner = SupervisedRunner(
    device=device,
    input_key="images",
    output_key=("logit_grapheme_root", "logit_vowel_diacritic", "logit_consonant_diacritic"),
    input_target_key=("grapheme_roots", "vowel_diacritics", "consonant_diacritics"),
)


# In[ ]:


runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    
    # our dataloaders
    loaders=loaders,
    
    callbacks=[
        # Each criterion is calculated separately.
        # Criterion for the grapheme root head. Select `criterion_key` to determine which loss you want to use for this head
        # It is similar to another heads.
        CriterionCallback(
            input_key="grapheme_roots",
            output_key="logit_grapheme_root",
            criterion_key='ce',
            prefix='loss_gr',
            multiplier=2.0,
        ),
        CriterionCallback(
            input_key="vowel_diacritics",
            output_key="logit_vowel_diacritic",
            criterion_key='ce',
            prefix='loss_wd',
            multiplier=1.0,
        ),
        CriterionCallback(
            input_key="consonant_diacritics",
            output_key="logit_consonant_diacritic",
            criterion_key='ce',
            prefix='loss_cd',
            multiplier=1.0,
        ),

        # And only then we aggregate everything into one loss.
        # Actually you can compute weighted loss, but the catalyst version should be 19.12.1.
        CriterionAggregatorCallback(
            prefix="loss",
            loss_aggregate_fn="sum", # It can be "sum", "weighted_sum" or "mean" in 19.12.1 version
            loss_keys=['loss_gr', 'loss_wd', 'loss_cd']
            # because we want weighted sum, we need to add scale for each loss
#             loss_keys={"loss_gr": 1.0, "loss_wd": 1.0, "loss_cd": 1.0},
        ),
        
        # metrics
        HMacroAveragedRecall(),
    ],
    # path to save logs
    logdir=logdir,
    
    num_epochs=num_epochs,
    
    # save our best checkpoint by IoU metric
    main_metric="hmar",
    # IoU needs to be maximized.
    minimize_metric=False,
    
    # for FP16. It uses the variable from the very first cell
    fp16=None,
    
    # for external monitoring tools, like Alchemy
    monitoring_params=None,
    
    # prints train logs
    verbose=True,
)


# # Unfreeze

# Now unfreeze the model and train with different settings. It is upto you !.
# Uncomment this cell to continue

# In[ ]:


from catalyst.contrib.schedulers import OneCycleLRWithWarmup


# In[ ]:


model.unfreeze()
logdir = "./logs/bengaliai/"

learning_rate = 0.0001
num_epochs = 15

optimizer = optim.AdamW(
    model.parameters(), 
    lr=learning_rate
)
scheduler = OneCycleLRWithWarmup(
    optimizer,
    num_steps=num_epochs,
    lr_range=[0.0005, 0.00001],
    warmup_steps=5,
    momentum_range=[0.85, 0.95]
) # Hack

runner = SupervisedRunner(
    device=device,
    input_key="images",
    output_key=("logit_grapheme_root", "logit_vowel_diacritic", "logit_consonant_diacritic"),
    input_target_key=("grapheme_roots", "vowel_diacritics", "consonant_diacritics"),
)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    
    # our dataloaders
    loaders=loaders,
    
    callbacks=[
        # Each criterion is calculated separately.
        # Criterion for the grapheme root head. Select `criterion_key` to determine which loss you want to use for this head
        # It is similar to another heads.
        CriterionCallback(
            input_key="grapheme_roots",
            output_key="logit_grapheme_root",
            criterion_key='ce',
            prefix='loss_gr',
            multiplier=2.0,
        ),
        CriterionCallback(
            input_key="vowel_diacritics",
            output_key="logit_vowel_diacritic",
            criterion_key='ce',
            prefix='loss_wd',
            multiplier=1.0,
        ),
        CriterionCallback(
            input_key="consonant_diacritics",
            output_key="logit_consonant_diacritic",
            criterion_key='ce',
            prefix='loss_cd',
            multiplier=1.0,
        ),

        # And only then we aggregate everything into one loss.
        # Actually you can compute weighted loss, but the catalyst version should be 19.12.1.
        CriterionAggregatorCallback(
            prefix="loss",
            loss_aggregate_fn="sum", # It can be "sum", "weighted_sum" or "mean" in 19.12.1 version
            loss_keys=['loss_gr', 'loss_wd', 'loss_cd']
            # because we want weighted sum, we need to add scale for each loss
#             loss_keys={"loss_gr": 1.0, "loss_wd": 1.0, "loss_cd": 1.0},
        ),
        
        # metrics
        HMacroAveragedRecall(),
    ],
    # path to save logs
    logdir=logdir,
    
    num_epochs=num_epochs,
    
    # save our best checkpoint by IoU metric
    main_metric="hmar",
    # IoU needs to be maximized.
    minimize_metric=False,
    
    # for FP16. It uses the variable from the very first cell
    fp16=None,
    
    # for external monitoring tools, like Alchemy
    monitoring_params=None,
    
    # prints train logs
    verbose=True,
)


# In[ ]:





# # Take your weights and do inference

# In[ ]:




