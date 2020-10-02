#!/usr/bin/env python
# coding: utf-8

# # Bengali.AI albumentations data augmentation tutorial
# 
# For CNN training, data augmentation is important to improve test accuracy (generalization performance). I will show some image preprocessing to increase the data variety.<br>
# **albumentations** library, *fast image augmentation library and easy to use wrapper around other libraries*, can be used for many kinds of data augmentation.
# 
# I will introduce several methods, especially useful for this competition.
# 
# Reference
#  - https://github.com/albumentations-team/albumentations
#  - https://arxiv.org/abs/1809.06839

# # Table of Contents:
# **[Fast data loading with feather](#load)**<br>
# **[Dataset](#dataset)**<br>
# **[How to apply albumentations augmentations](#apply)**<br>
# **[Blur Related Methods](#blur)**<br>
# **[Noise Related Methods](#noise)**<br>
# **[Cutout Related Methods](#cutout)**<br>
# **[Distortion Related Methods](#distortion)**<br>
# **[Brightness, contrast Related Methods](#brightness)**<br>
# **[Affine Related Methods](#affine)**<br>
# **[Reference and further reading](#ref)**<br>

# In[ ]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm.notebook import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- setup ---
pd.set_option('max_columns', 50)


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


debug=True
submission=False
batch_size=32
device='cuda:0'
out='.'
image_size=64
ncol=6


# In[ ]:


datadir = Path('/kaggle/input/bengaliai-cv19')
featherdir = Path('/kaggle/input/bengaliaicv19feather')
outdir = Path('.')


# <a id="load"></a>
# # Fast data loading with feather
# 
# Refer [Bengali.AI super fast data loading with feather](https://www.kaggle.com/corochann/bengali-ai-super-fast-data-loading-with-feather) and [dataset](https://www.kaggle.com/corochann/bengaliaicv19feather) for detail.<br/>
# Original `parquet` format takes about 60 sec to load 1 data, while `feather` format takes about **2 sec to load 1 data!!!**
# 
# ### How to add dataset
# 
# When you write kernel, click "+ Add Data" botton on right top.<br/>
# Then inside window pop-up, you can see "Search Datasets" text box on right top.<br/>
# You can type "bengaliai-cv19-feather" to find this dataset and press "Add" botton to add the data.

# In[ ]:


import numpy as np
import pandas as pd
import gc

def prepare_image(datadir, featherdir, data_type='train',
                  submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    if submission:
        image_df_list = [pd.read_parquet(datadir / f'{data_type}_image_data_{i}.parquet')
                         for i in indices]
    else:
        image_df_list = [pd.read_feather(featherdir / f'{data_type}_image_data_{i}.feather')
                         for i in indices]

    print('image_df_list', len(image_df_list))
    HEIGHT = 137
    WIDTH = 236
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    return images


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_csv(datadir/'train.csv')\ntrain_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values\nindices = [0] if debug else [0, 1, 2, 3]\ntrain_images = prepare_image(\n    datadir, featherdir, data_type='train', submission=False, indices=indices)")


# <a id="dataset"></a>
# # Dataset

# This `DatasetMixin` class can be used to define any custom dataset class in pytorch. We can implement `get_example(self, i)` method to return `i`-th data.
# 
# Here I return `i`-th image `x` and `label`, with scaling image to be value ranges between 0~1.

# In[ ]:


"""
Referenced `chainer.dataset.DatasetMixin` to work with pytorch Dataset.
"""
import numpy
import six
import torch
from torch.utils.data.dataset import Dataset


class DatasetMixin(Dataset):

    def __init__(self, transform=None):
        self.transform = transform

    def __getitem__(self, index):
        """Returns an example or a sequence of examples."""
        if torch.is_tensor(index):
            index = index.tolist()
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example_wrapper(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, numpy.ndarray):
            return [self.get_example_wrapper(i) for i in index]
        else:
            return self.get_example_wrapper(index)

    def __len__(self):
        """Returns the number of data points."""
        raise NotImplementedError

    def get_example_wrapper(self, i):
        """Wrapper of `get_example`, to apply `transform` if necessary"""
        example = self.get_example(i)
        if self.transform:
            example = self.transform(example)
        return example

    def get_example(self, i):
        """Returns the i-th example.

        Implementations should override it. It should raise :class:`IndexError`
        if the index is invalid.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        raise NotImplementedError


# In[ ]:


import numpy as np


class BengaliAIDataset(DatasetMixin):
    def __init__(self, images, labels=None, transform=None, indices=None):
        super(BengaliAIDataset, self).__init__(transform=transform)
        self.images = images
        self.labels = labels
        if indices is None:
            indices = np.arange(len(images))
        self.indices = indices
        self.train = labels is not None

    def __len__(self):
        """return length of this dataset"""
        return len(self.indices)

    def get_example(self, i):
        """Return i-th data"""
        i = self.indices[i]
        x = self.images[i]
        # Opposite white and black: background will be white and
        # for future Affine transformation
        x = (255 - x).astype(np.float32) / 255.
        if self.train:
            y = self.labels[i]
            return x, y
        else:
            return x


# In[ ]:


train_dataset = BengaliAIDataset(train_images, train_labels)                                 


# ## Original data
# 
# Let's see original data at first

# In[ ]:


nrow, ncol = 1, 6

fig, axes = plt.subplots(nrow, ncol, figsize=(20, 2))
axes = axes.flatten()
for i, ax in enumerate(axes):
    image, label = train_dataset[i]
    ax.imshow(image, cmap='Greys')
    ax.set_title(f'label: {label}')
plt.tight_layout()


# <a id="apply"></a>
# ## How to apply albumentations augmentations
# 
# When we have `image` array, we can apply albumentations augmentation by calling **`aug(image=image)['image']`**, where `aug` is various methods implemented in `albumentations`.
# 
# Let's see example, I will apply `aug = A.Blur(p=1.0)`. You can see that the image is blurred from original image.

# In[ ]:


import albumentations as A

aug = A.Blur(p=1.0)

nrow, ncol = 1, 6

fig, axes = plt.subplots(nrow, ncol, figsize=(20, 2))
axes = axes.flatten()
for i, ax in enumerate(axes):
    image, label = train_dataset[i]
    # I added only this 1 line!
    image = aug(image=image)['image']
    ax.imshow(image, cmap='Greys')
    ax.set_title(f'label: {label}')
plt.tight_layout()


# In[ ]:


def show_images(aug_dict, ncol=6):
    nrow = len(aug_dict)

    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 2 * nrow), squeeze=False)
    for i, (key, aug) in enumerate(aug_dict.items()):
        for j in range(ncol):
            ax = axes[i, j]
            if j == 0:
                ax.text(0.5, 0.5, key, horizontalalignment='center', verticalalignment='center', fontsize=15)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.axis('off')
            else:
                image, label = train_dataset[j-1]
                if aug is not None:
                    image = aug(image=image)['image']
                ax.imshow(image, cmap='Greys')
                ax.set_title(f'label: {label}')
    plt.tight_layout()
    plt.show()
    plt.close()


# Many methods are supported in albumentations, let's see each methods.<br>
# I categorized methods by section so that you can refer easily :)

# <a id="blur"></a>
# # Blur related methods
# 
#  - A.Blur
#  - A.MedianBlur
#  - A.GaussianBlur
#  - A.MotionBlur

# In[ ]:


show_images({'Original': None,
             'Blur': A.Blur(p=1.0),
             'MedianBlur': A.MedianBlur(blur_limit=5, p=1.0),
             'GaussianBlur': A.GaussianBlur(p=1.0),
             'MotionBlur': A.MotionBlur(p=1.0)},
            ncol=ncol)


# <a id="noise"></a>
# # Noise related methods
# 
#  - A.GaussNoise
#  - A.MultiplicativeNoise

# In[ ]:


show_images({'Original': None,
             'GaussNoise': A.GaussNoise(var_limit=5. / 255., p=1.0),
             'MultiplicativeNoise': A.MultiplicativeNoise(p=1.0)},
            ncol=ncol)


# <a id="cutout"></a>
# # Cutout related methods
# 
# It adds square sized mask to images for data augmentation. CNN need to learn target label by "watching" part of the images.<br>
# Refer: https://arxiv.org/abs/1708.04552
# 
#  - A.Cutout <-- It is deprecated.
#  - A.CoarseDropout

# In[ ]:


show_images({'Original': None,
             'Cutout':A.Cutout(num_holes=8,  max_h_size=20, max_w_size=20, p=1.0),
             'CoarseDropout': A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=1.0)},
            ncol=ncol)


# <a id="distortion"></a>
# # Distortion related methods
# 
#  - A.GridDistortion
#  - A.ElasticTransform: Refer http://cognitivemedium.com/assets/rmnist/Simard.pdf

# In[ ]:


show_images({'Original': None,
             'GridDistortion':A.GridDistortion(p=1.0),
             'ElasticTransform': A.ElasticTransform(sigma=50, alpha=1, alpha_affine=10, p=1.0)},
            ncol=ncol)


# <a id="brightness"></a>
# # Brightness, contrast related methods
# 
#  - A.RandomBrightness <-- Deprecated
#  - A.RandomContrast <-- Deprecated
#  - A.RandomBrightnessContrast

# In[ ]:


show_images({'Original': None,
             'RandomBrightness': A.RandomBrightness(p=1.0),
             'RandomContrast': A.RandomContrast(p=1.0),
             'RandomBrightnessContrast': A.RandomBrightnessContrast(p=1.0)},
            ncol=ncol)


# <a id="affine"></a>
# # Affine related methods
# 
#  - A.RandomBrightness <-- Deprecated
#  - A.RandomContrast <-- Deprecated
#  - A.RandomBrightnessContrast

# In[ ]:


show_images({'Original': None,
             'IAAPiecewiseAffine': A.IAAPiecewiseAffine(p=1.0),
             'ShiftScaleRotate': A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=30,
                p=1.0)},
            ncol=ncol)


# <a id="ref"></a>
# # Reference and further reading
# 
# That's all for the tutorial of this kernel. Below are the next reading contents.<br>
# Especially, I will write training code based on this data augmentation in this kernel: **[Bengali: SEResNeXt prediction with pytorch](https://www.kaggle.com/corochann/bengali-seresnext-prediction-with-pytorch)**.
# 
# #### Kernel
# 
# **[Bangali.AI super fast data loading with feather](https://www.kaggle.com/corochann/bangali-ai-super-fast-data-loading-with-feather)**<br>
# Simple example of how use feather format data to load data faster.
# 
# **[Bengali: SEResNeXt prediction with pytorch](https://www.kaggle.com/corochann/bengali-seresnext-prediction-with-pytorch)**<br>
# **Training code using this kernel's data augmentation, please check this too!**
# 
# **[Bengali: SEResNeXt prediction with pytorch](https://www.kaggle.com/corochann/bengali-seresnext-prediction-with-pytorch)**<br>
# Prediction code of above trained model.
# 
# **[Deep learning - CNN with Chainer: LB 0.99700](https://www.kaggle.com/corochann/deep-learning-cnn-with-chainer-lb-0-99700)**<br>
# Data augmentation idea is based on this kernel, which achieves quite high accuracy on MNIST task.
# 
# #### Dataset
# **[bengaliai-cv19-feather](https://www.kaggle.com/corochann/bengaliaicv19feather)**<br>
# Feather format dataset
# 
# **[bengaliaicv19_seresnext101_32x4d](https://www.kaggle.com/corochann/bengaliaicv19-seresnext101-32x4d)**<br>
# Trained model weight
# 
# **[bengaliaicv19_trainedmodels](https://www.kaggle.com/corochann/bengaliaicv19-trainedmodels)**<br>
# Trained model weight
# 
# #### Library
# **https://github.com/albumentations-team/albumentations**
# 
# fast image augmentation library and easy to use wrapper around other libraries https://arxiv.org/abs/1809.06839<br>
# I could not show all the methods, you can find more methods in the library, check yourself!

# <h3 style="color:red">If this kernel helps you, please upvote to keep me motivated :)<br>Thanks!</h3>
