#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


print(os.listdir("../input/pixabay/"))


# In[ ]:


import fastai
from fastai.imports import *
from fastai.vision import *
from fastai.metrics import *
from fastai.gen_doc.nbdoc import *
print('fast.ai version:{}'.format(fastai.__version__))


# In[ ]:


#model = models.resnet152
model = models.densenet201
#model.load_state_dict(torch.load('../input/pretrained-pytorch-dog-and-cat-models/resnet50.pth'))
WORK_DIR = os.getcwd()
IMAGE_DIR = Path('../input/')
image_size=224
batch_size=32


# In[ ]:


labels_df = pd.read_csv('../input/labels.csv')


# In[ ]:


fns = []
ids = []
root = '../input/pixabay/'
for t in ['cats/','dogs/']:
    for l in ['0/','1/']:
        i_paths = os.listdir(f'../input/pixabay/{t}{l}')
        fns += [root + t + l + i for i in i_paths]
        ids += [p[:-4] for p in i_paths]


# In[ ]:


len(fns), len(ids)


# In[ ]:


labels_df = labels_df.set_index('id')


# In[ ]:


a = labels_df.loc[ids]


# In[ ]:


labels = a['cute'].values


# In[ ]:


data = ImageDataBunch.from_lists(path = '',fnames = fns, labels = labels,ds_tfms=get_transforms(), 
                                   test ='test',
                                   size=image_size, 
                                   bs=batch_size,
                                   num_workers=0).normalize(imagenet_stats)


# In[ ]:


data


# In[ ]:


learn = create_cnn(data, model, metrics=accuracy, model_dir=WORK_DIR)


# In[ ]:


learn.fit_one_cycle(2)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.load('stage-1')
learn.unfreeze()
learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-2')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9)


# In[ ]:




