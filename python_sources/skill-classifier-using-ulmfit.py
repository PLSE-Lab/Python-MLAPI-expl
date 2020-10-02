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


data = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
data.head()


# In[ ]:


data.dropna(inplace = True)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


from pathlib import Path
from fastai.text import *


# In[ ]:


fast_data = data.sample(frac=1).reset_index(drop=True)
fast_data.shape[0]*.7


# In[ ]:


train_df, valid_df = fast_data.loc[:115330,:],fast_data.loc[115330:,:]


# In[ ]:


train_df.head()


# In[ ]:


path =Path(".")


# In[ ]:


train_df


# In[ ]:


data_lm = TextLMDataBunch.from_df(path, train_df, valid_df, text_cols=['review'], bs=32)
data_clas = TextClasDataBunch.from_df(path, train_df, valid_df, text_cols=['review'], label_cols=['sentiment'], bs=64)


# In[ ]:


data_lm.show_batch()


# In[ ]:


data_clas.show_batch()


# In[ ]:


learn = language_model_learner(data_lm, arch = AWD_LSTM, pretrained = True, drop_mult=0.4)
learn.lr_find() # find learning rate
learn.recorder.plot() # plot learning rate graph


# In[ ]:


learn.fit_one_cycle(2, 1e-2)


# In[ ]:


learn.unfreeze() # must be done before calling lr_find
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, 1e-3)


# In[ ]:


learn.save_encoder('word-enc')


# In[ ]:


learn = text_classifier_learner(data_clas, arch = AWD_LSTM, pretrained = True, drop_mult=0.3)
learn.load_encoder('word-enc')

# find and plot learning rate
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(1, 1e-2)

# unfreeze one layer group and train another epoch
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))

learn.unfreeze()
learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))


# In[ ]:


learn.export(file = Path("/kaggle/working/export.pkl"))


# In[ ]:


print(learn.predict("Engineer")[0])


# In[ ]:




