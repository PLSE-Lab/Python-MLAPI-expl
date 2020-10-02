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


# Any results you write to the current directory are saved as output.


# In[ ]:


import os
from fastai.vision import *
from fastai import *
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from tqdm.notebook import tqdm
import gc
from pylab import imread,subplot,imshow,show
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/output'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


path = "/kaggle/input/ewaste-dataset"


# In[ ]:


size = 224
bs=6


# In[ ]:


size = 224
bs = 64
data = ImageDataBunch.from_folder(path, 
                                  ds_tfms=get_transforms(max_rotate=0.1,max_lighting=0.15),
                                  valid_pct=0.2, 
                                  size=size, 
                                  bs=bs)


# In[ ]:


data.show_batch(rows=4)


# In[ ]:


fb = FBeta()
fb.average='macro'


# In[ ]:





# In[ ]:


get_ipython().system('git clone https://github.com/fastai/fastai2')


# In[ ]:


cd fastai2


# In[ ]:


pip install -e ".[dev]"


# In[ ]:


from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


learner = cnn_learner(data, models. resnet50, metrics=[accuracy])


# In[ ]:


learner.fit_one_cycle(5,1e-3)


# In[ ]:





# In[ ]:


learner.save('model_ewaste')


# In[ ]:


learner.export(file = Path("/kaggle/working/export.pkl"))


# In[ ]:


model_dir="/tmp/model/"


# In[ ]:


learner.model_dir='/kaggle/working/'


# In[ ]:


img = open_image("/kaggle/input/ewaste-dataset/mobile/456.jpeg")
print(learner.predict(img)[0])


# In[ ]:


deployed_path = "kaggle/working"


# In[ ]:


learn = load_learner(deployed_path)


# In[ ]:




