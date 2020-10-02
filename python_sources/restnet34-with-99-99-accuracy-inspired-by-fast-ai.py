#!/usr/bin/env python
# coding: utf-8

# #Inspired by fast.ai Practical Deep Learning for Coders(v3) lession 1

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


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


bs = 64


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir('../input/fruits-360_dataset/fruits-360'))


# In[ ]:


data_dir='../input/fruits-360_dataset/fruits-360'

list = os.listdir(data_dir) 
number_files = len(list)
print(number_files)


# In[ ]:


path=Path(data_dir)
path


# In[ ]:


data = ImageDataBunch.from_folder(path,  
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(do_flip=True,flip_vert=False, max_rotate=90),
                                  size=224,bs=64, 
                                  num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(10,10))


# In[ ]:


print(data.classes)
len(data.classes)


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(3)


# In[ ]:


learn.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(30,25))


# In[ ]:


interp.most_confused(min_val=1)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(3)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:




