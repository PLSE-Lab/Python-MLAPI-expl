#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

from sklearn.metrics import confusion_matrix
from fastai import *
from fastai.vision import *

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm_notebook
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

print(os.listdir("../input"))


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv')\ntest = pd.read_csv('../input/test.csv')")


# In[ ]:


get_ipython().system('ls ../input/train_images')


# In[ ]:


train.head()


# In[ ]:


x_train = train['id_code']
y_train = train['diagnosis']


# In[ ]:


tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.1,max_lighting=0.1,p_lighting=0.5)


# In[ ]:


bs = 64 #smaller batch size is better for training, but may take longer
sz=224
n_splits=5


# In[ ]:


def get_data(train_index, valid_index, bs=bs, sz=sz):
    src = (ImageList.from_df(df=train, path='../input/train_images', cols='id_code', suffix='.png') #get dataset from dataset
            .split_by_idxs(train_idx=train_index, valid_idx=valid_index) #Splitting the dataset
            .label_from_df(cols='diagnosis') #obtain labels from the level column
          )
    data= (src.transform(tfms, size=sz, resize_method=ResizeMethod.SQUISH, padding_mode='zeros') #Data augmentation
            .databunch(bs=bs, num_workers=2) #DataBunch
            .normalize(imagenet_stats) #Normalize     
           )
    return data


# In[ ]:


skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=620402)


# In[ ]:


def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.argmax(y_hat,1), y, weights='quadratic'),device='cuda:0')


# In[ ]:


get_ipython().run_cell_magic('time', '', "model_num = 0\ndata = None\nlearn = None\nfor train_index, valid_index in tqdm_notebook(skf.split(x_train, y_train)):\n    if data is not None:\n        del data\n    data = get_data(train_index = train_index, valid_index = valid_index)\n    print(data)\n    if learn is not None:\n        learn.destroy()\n        del learn\n    learn = cnn_learner(data, models.resnet50, metrics=[accuracy, quadratic_kappa]).mixup().to_fp16()\n    learn.freeze()\n    learn.fit_one_cycle(3, slice(1e-2))\n    learn.unfreeze()\n    learn.fit_one_cycle(8, slice(1e-5,1e-3))\n    learn.path=Path('.')\n    learn.export(f'./model_{model_num}.pkl', destroy=True)\n    model_num += 1")

