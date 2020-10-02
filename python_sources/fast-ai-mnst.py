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


get_ipython().system('pip3 install git+https://github.com/fastai/fastai.git')


# In[ ]:


from fastai.vision import * 
from fastai import *


# In[ ]:


path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.fit(1)


# In[ ]:


learn.recorder.plot_lr(show_moms=True)


# In[ ]:


ds = data.train_ds


# In[ ]:


img,label = ds[0]
img


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(6,6))


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


img = learn.data.train_ds[0][0]
learn.predict(img)


# In[ ]:




