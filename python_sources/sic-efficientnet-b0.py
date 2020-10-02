#!/usr/bin/env python
# coding: utf-8

# # SHOKUNIN 2019: Synthetic Image Classification
# 
# This kernel solves the TW Synthetic Image Classification problem using [EfficientNet-B0](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html) trained from scratch using [1cycle policy](https://arxiv.org/pdf/1803.09820.pdf) training and the Fast.ai library on Pytorch.
# 
# Current gets > 0.94 on the LB with an underfitted model.

# In[ ]:


import sys
import random
import numpy as np
import os

from tqdm import tqdm_notebook as tqdm
import torch
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

from fastai.vision.data import ImageDataBunch, DatasetType, ImageList, ResizeMethod
from fastai.vision.transform import get_transforms
from fastai.vision.learner import Learner
from fastai.metrics import accuracy
from fastai.train import ClassificationInterpretation
from PIL import Image

sys.path.append('../input/efficientnet-pytorch/efficientnet-pytorch/EfficientNet-PyTorch-master')

import efficientnet_pytorch


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

augs = get_transforms(
    do_flip=True,
    max_rotate=10.,
    max_zoom=1.1,
    max_lighting=0.2,
    max_warp=0.2,
    p_affine=0.75)


# In[ ]:


seed_everything(SEED)


# ## Create dataset

# In[ ]:


data = ImageDataBunch.from_folder(
    '../',
    train='../input/synthetic-image-classification/synimg/train', valid_pct=0.2,
    ds_tfms=augs, seed=SEED,
    classes=CLASSES, size=64, test=None, resize_method=ResizeMethod.PAD, padding_mode='zeros')


# In[ ]:


assert len(data.classes) == len(CLASSES)


# In[ ]:


test_df = pd.read_csv('../input/synthetic-image-classification/synimg/synimg/test/data_nostyle.csv')[['filepath']]


# In[ ]:


data.add_test(ImageList.from_df(test_df, '../input/synthetic-image-classification/synimg/')) 


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
for p in Path('../input/synthetic-image-classification/synimg/synimg/train').iterdir():
    if not p.is_dir():
        continue

    for f in p.iterdir():
        all_files.append(f)


# In[ ]:


assert len(all_files) == len(pd.read_csv('../input/synthetic-image-classification/synimg/synimg/train/data.csv'))


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


pd.read_csv('../input/synthetic-image-classification/synimg/synimg/train/data.csv').style_name.value_counts().plot.bar()


# ## Model training

# In[ ]:


model = efficientnet_pytorch.EfficientNet.from_name(
    'efficientnet-b0', override_params={'num_classes': len(CLASSES)})


# In[ ]:


learn = Learner(data, model, metrics=[accuracy])


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


# For testing
# learn.fit_one_cycle(1, 2e-02)


# In[ ]:


learn.fit_one_cycle(20, 2e-02)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.validate()


# In[ ]:


learn.fit_one_cycle(4, 2e-3)


# In[ ]:


learn.validate()


# In[ ]:


learn.recorder.plot_losses()


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


test_preds, _ = learn.TTA(ds_type=DatasetType.Test)


# In[ ]:


pred_names = [data.classes[i] for i in test_preds.argmax(1)]


# In[ ]:


output_df = pd.read_csv('../input/synthetic-image-classification/sample_submission.csv')


# In[ ]:


output_df['style_name'] = pred_names


# In[ ]:


output_df.head()


# In[ ]:


output_df.to_csv('submission.csv', index=False)


# In[ ]:


pd.read_csv('submission.csv')

