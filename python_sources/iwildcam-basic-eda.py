#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# Camera Traps (or Wild Cams) enable the automatic collection of large quantities of image data. Biologists all over the world use camera traps to monitor biodiversity and population density of animal species
# As biologists try to expand the scope of these models from specific regions where they have collected training data to nearby areas they are faced with an interesting probem: how do you classify a species in a new region that you may not have seen in previous training data?
# 
# In this challenge the training data and test data are from different regions, namely The American Southwest and the American Northwest.
# 
# We have 23 classes for which we need to make a classification. Using external data is allowed.
# 
# ![](https://camo.githubusercontent.com/3398bff8743feb9e94ffbeae1c2c32cdbda97b20/68747470733a2f2f7261776769742e636f6d2f7669736970656469612f6977696c6463616d5f636f6d702f736172612f6173736574732f6977696c6463616d5f323031395f62616e6e65722e6a7067)

# In[ ]:


# libraries
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time 
from PIL import Image
train_on_gpu = True
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import accuracy_score
import cv2

import altair as alt
from  altair.vega import v3
from IPython.display import HTML
import json


# In[ ]:


get_ipython().system('pip install albumentations > /dev/null 2>&1')
get_ipython().system('pip install pretrainedmodels > /dev/null 2>&1')
get_ipython().system('pip install kekas > /dev/null 2>&1')
get_ipython().system('pip install adabound > /dev/null 2>&1')


# In[ ]:


import albumentations
from albumentations import torch as AT
import pretrainedmodels
import adabound

from kekas import Keker, DataOwner, DataKek
from kekas.transformations import Transformer, to_torch, normalize
from kekas.metrics import accuracy
from kekas.modules import Flatten, AdaptiveConcatPool2d
from kekas.callbacks import Callback, Callbacks, DebuggerCallback
from kekas.utils import DotDict


# In[ ]:


# Preparing altair. I use code from this great kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey

vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION
vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
noext = "?noext"

paths = {
    'vega': vega_url + noext,
    'vega-lib': vega_lib_url + noext,
    'vega-lite': vega_lite_url + noext,
    'vega-embed': vega_embed_url + noext
}

workaround = """
requirejs.config({{
    baseUrl: 'https://cdn.jsdelivr.net/npm/',
    paths: {}
}});
"""

#------------------------------------------------ Defs for future rendering
def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay outside and 
    return wrapped
            
@add_autoincrement
def render(chart, id="vega-chart"):
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vg_embed) {{
        const spec = {chart};     
        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
        console.log("anything?");
    }});
    console.log("really...anything?");
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )

HTML("".join((
    "<script>",
    workaround.format(json.dumps(paths)),
    "</script>",
)))


# ## Data overview

# In[ ]:


os.listdir('../input/')


# We have separate `csv` files for train and test data and folders with images.
# 
# Class names are presented as a string in the description of the competition, let's convert this into a dictionary.

# In[ ]:


classes = """empty, 0
deer, 1
moose, 2
squirrel, 3
rodent, 4
small_mammal, 5
elk, 6
pronghorn_antelope, 7
rabbit, 8
bighorn_sheep, 9
fox, 10
coyote, 11
black_bear, 12
raccoon, 13
skunk, 14
wolf, 15
bobcat, 16
cat, 17
dog, 18
opossum, 19
bison, 20
mountain_goat, 21
mountain_lion, 22""".split('\n')
classes = {int(i.split(', ')[1]): i.split(', ')[0] for i in classes}
classes


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')
train['classes'] = train['category_id'].apply(lambda x: classes[x])


# In[ ]:


train.head()


# Here we have timestamps, locations, rights holder, info about the sequence of photos and height/width.

# In[ ]:


train.classes.unique()


# And we have only 14 classes of 23 in train data. It seems that using external data is necessary. Let's see how far can we go without it.

# In[ ]:


fig = plt.figure(figsize=(25, 60))
imgs = [np.random.choice(train.loc[train['classes'] == i, 'file_name'], 4) for i in train.classes.unique()]
imgs = [i for j in imgs for i in j]
labels = [[i] * 4 for i in train.classes.unique()]
labels = [i for j in labels for i in j]
for idx, img in enumerate(imgs):
    ax = fig.add_subplot(14, 4, idx + 1, xticks=[], yticks=[])
    im = Image.open("../input/train_images/" + img)
    plt.imshow(im)
    ax.set_title(f'Label: {labels[idx]}')


# In[ ]:


target_count = train['classes'].value_counts().reset_index().rename(columns={'index': 'target'})
render(alt.Chart(target_count).mark_bar().encode(
    y=alt.Y("target:N", axis=alt.Axis(title='Surface'), sort=list(target_count['target'])),
    x=alt.X('classes:Q', axis=alt.Axis(title='Count')),
    tooltip=['target', 'classes']
).properties(title="Counts of target classes", width=400).interactive())


# We can see that working with this data is quite difficult. Class disbalance is huge and there are a lot of empty pictures. Also the photos themselves present a challenge - animals can be in a small part of the screen and be obscured.

# In[ ]:


lc = pd.crosstab(train['location'], train['classes']).reset_index()
lc = pd.melt(lc, 'location', lc.columns.tolist()[1:], 'class', 'count')
lc = lc.loc[lc['class'] != 'empty']
render(alt.Chart(lc).mark_circle().encode(
    y='location:N',
    x='class',
    size='count',
    tooltip=['location', 'class', 'count']
).properties(title="Animals and locations", width=400).interactive())

