#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:




from fastai import *
from fastai.vision import *


# In[ ]:


PATH = '/kaggle/input/100-bird-species'


# In[ ]:


get_ipython().run_line_magic('pinfo', 'ImageImageList')


# In[ ]:


data = (ImageList.from_folder(PATH) #Where to find the data? -> in path and its subfolders
        .split_by_folder()              #How to split in train/valid? -> use the folders
        .label_from_folder()            #How to label? -> depending on the folder of the filenames
        .databunch())


# In[ ]:


len(data.valid_dl)


# In[ ]:


429*64


# In[ ]:


data.normalize(imagenet_stats)


# In[ ]:


data.show_batch()


# In[ ]:


len(data.classes)


# In[ ]:


learn = create_cnn(data,models.resnet34,metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.data


# In[ ]:


learn.lr_find()


# In[ ]:


learn.model_dir='/kaggle/working/' 


# In[ ]:


learn.save('Stage-1')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(12,figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(20,20), dpi=60)


# In[ ]:


interp.most_confused()


# (Actual,Predicted,count)

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2,max_lr=slice(1e-5,1e-4))


# In[ ]:


data_test = (ImageList.from_folder(PATH)
             .split_by_folder(train='train',valid='test')
             .label_from_folder()
             .databunch()
             .normalize(imagenet_stats)
            )


# In[ ]:


learn.validate(data_test.valid_dl) #test set 
#Gives [loss, error_rate]
# accuracy = (1-error_rate)*100


# In[ ]:


learn.validate(data.valid_dl) #validation set


# In[ ]:




