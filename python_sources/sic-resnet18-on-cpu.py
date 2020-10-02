#!/usr/bin/env python
# coding: utf-8

# # SHOKUNIN 2019: Synthetic Image Classification
# 
# This kernel solves the TW Synthetic Image Classification problem using [ResNet18](https://arxiv.org/abs/1512.03385) and just a CPU,
# 
# It uses the Fast.ai library on Pytorch.
# 
# ## To do
# 
# * Convert to a library.
# * Add tests.
# * Try more epochs.
# * Try with a bigger sample size.

# In[ ]:


import sys
import random
import numpy as np
import os
from pathlib import Path

from tqdm import tqdm_notebook as tqdm
import torch
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

from fastai.vision.data import ImageDataBunch, DatasetType, ImageList, ResizeMethod
from fastai.vision.transform import get_transforms
from fastai.vision.learner import cnn_learner
from fastai.metrics import accuracy
from fastai.train import ClassificationInterpretation
from fastai.vision.models import resnet18
from PIL import Image


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ## Tunable params

# In[ ]:


CLASSES = [
    'Beijing', 'Brisbane', 'Geneva', 'HongKong', 'Luanda',
    'Melbourne', 'Seoul', 'Singapore', 'Sydney', 'Zurich']

SEED = 420
BS = 64
SAMPLE_SIZE = 10_000
NUM_EPOCHS = 1
MODEL_ARCH = resnet18

augs = get_transforms(
    do_flip=True,
    max_rotate=10.,
    max_zoom=1.1,
    max_lighting=0.2,
    max_warp=0.2,
    p_affine=0.75)

DATA_DIR = Path('../input/synthetic-image-classification/synimg/')
SAMPLE_SUB_PATH = Path('../input/synthetic-image-classification/sample_submission.csv')
WORKING_DIR = Path('./')


# In[ ]:


seed_everything(SEED)


# ## Create dataset

# In[ ]:


train_df = pd.read_csv(DATA_DIR/'synimg/train/data.csv')[['filepath', 'style_name']]

# Kaggle thing
train_df.filepath = str(DATA_DIR) + '/' + train_df.filepath

if SAMPLE_SIZE:
    train_df = train_df.sample(n=SAMPLE_SIZE, random_state=SEED)
    
test_df = pd.read_csv(DATA_DIR/'synimg/test/data_nostyle.csv')[['filepath']]


# In[ ]:


data = ImageDataBunch.from_df(
    WORKING_DIR,
    df=train_df, valid_pct=0.2,
    ds_tfms=augs, seed=SEED, size=64, test=None, resize_method=ResizeMethod.PAD, padding_mode='zeros')


# In[ ]:


data.add_test(ImageList.from_df(test_df, DATA_DIR)) 


# In[ ]:


assert len(data.classes) == len(CLASSES)


# ## EDA

# ### Image examples 

# #### Train/validation set

# In[ ]:


data.show_batch()


# In[ ]:


data.show_batch(ds_type=DatasetType.Valid)


# #### Test set

# In[ ]:


data.show_batch(ds_type=DatasetType.Test)


# ### Size distribution

# In[ ]:


all_files = []
for p in Path(DATA_DIR/'synimg/train').iterdir():
    if not p.is_dir():
        continue

    for f in p.iterdir():
        all_files.append(f)


# In[ ]:


sizes = [Image.open(f).size for f in tqdm(all_files)]


# In[ ]:


heights = [s[0] for s in sizes]
widths = [s[1] for s in sizes]


# In[ ]:


plt.hist(heights)
plt.title('Height distribution')
plt.show()


# In[ ]:


plt.hist(widths)
plt.title('Width distribution')
plt.show()


# ### Class balance

# In[ ]:


pd.read_csv(DATA_DIR/'synimg/train/data.csv').style_name.value_counts().plot.bar()


# ## Model training

# In[ ]:


learn = cnn_learner(data, resnet18, metrics=[accuracy])


# In[ ]:


learn.fit_one_cycle(NUM_EPOCHS, 3e-03)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.validate()


# ## Interpret results

# ### Top losses

# In[ ]:


interpretation = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interpretation.plot_top_losses(12, figsize=(15, 12))


# ### Confusion matrix

# In[ ]:


interpretation.plot_confusion_matrix(figsize=(8, 8), dpi=60)


# ## Create submission

# In[ ]:


test_preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


pred_names = [data.classes[i] for i in test_preds.argmax(1)]


# In[ ]:


output_df = pd.read_csv(SAMPLE_SUB_PATH)


# In[ ]:


output_df['style_name'] = pred_names


# In[ ]:


output_df.head()


# In[ ]:


output_df.to_csv(WORKING_DIR/'submission.csv', index=False)

