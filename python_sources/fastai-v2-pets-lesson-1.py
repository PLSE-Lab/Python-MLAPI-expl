#!/usr/bin/env python
# coding: utf-8

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


get_ipython().system('pip install torch torchvision feather-format kornia pyarrow --upgrade   > /dev/null')
get_ipython().system('pip install git+https://github.com/fastai/fastai_dev                    > /dev/null')


# In[ ]:


from fastai2.basics           import *
from fastai2.vision.all       import *
from fastai2.medical.imaging  import *
from fastai2.callback.tracker import *
from fastai2.callback.all     import *

np.set_printoptions(linewidth=120)
matplotlib.rcParams['image.cmap'] = 'bone'


# In[ ]:


bs = 64


# In[ ]:


path = untar_data(URLs.PETS); path


# In[ ]:


path.ls()


# In[ ]:


path_anno = path/'annotations'
path_img = path/'images'


# In[ ]:


fnames = get_image_files(path_img)
fnames[:5]


# In[ ]:


np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'


# In[ ]:


dbunch = ImageDataBunch.from_name_re(path, fnames, pat, item_tfms=RandomResizedCrop(460, min_scale=0.75), bs=bs,
                                     batch_tfms=[*aug_transforms(size=224, max_warp=0), Normalize(*imagenet_stats)])


# In[ ]:


dbunch.show_batch(max_n=9, figsize=(10,10))


# In[ ]:


print(dbunch.vocab)
len(dbunch.vocab),dbunch.c


# In[ ]:


learn = cnn_learner(dbunch, resnet34, metrics=error_rate).to_fp16()


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.recorder.plot_loss()


# In[ ]:


learn.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(dbunch.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.fit_one_cycle(2, lr_max = slice(1e-6, 1e-4))


# # ResNet 50

# In[ ]:


dbunch = ImageDataBunch.from_name_re(path_img, fnames, pat, item_tfms=RandomResizedCrop(460, min_scale=0.75), bs=bs//2,
                                     batch_tfms=[*aug_transforms(size=299, max_warp=0), Normalize(*imagenet_stats)])


# In[ ]:


learn = cnn_learner(dbunch, resnet50, metrics=error_rate).to_fp16()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.fit_one_cycle(8)


# In[ ]:


learn.save('stage-1-50')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, lr_max=slice(1e-6,1e-4))


# In[ ]:


learn.load('stage-1-50');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

