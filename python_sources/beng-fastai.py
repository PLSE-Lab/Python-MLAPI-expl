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
from fastai.vision import *

# Any results you write to the current directory are saved as output.


# In[ ]:


# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# In[ ]:


path="/kaggle/input/grapheme-imgs-128x128"
grapheme_root="/kaggle/input/beng-labels/beng_labels/grapheme_root.csv"
vowel_diacritic="/kaggle/input/beng-labels/beng_labels/vowel_diacritic.csv"
consonant_diacritic="/kaggle/input/beng-labels/beng_labels/consonant_diacritic.csv"


# In[ ]:


doc(ImageDataBunch.from_csv)


# In[ ]:


#tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
data = ImageDataBunch.from_csv(path, csv_labels=grapheme_root,valid_pct=0.2 , size=128,suffix='.png',bs=64)
data.normalize()
data.show_batch(rows=3, figsize=(6,6))


# In[ ]:


learn = cnn_learner(data, models.densenet121, metrics=accuracy)
learn.model_dir = "/kaggle/working" 
learn.save("stage-1")


# In[ ]:


# learn.unfreeze


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(80,max_lr=0.01)
learn.recorder.plot_losses()

