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


# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from fastai import *
from fastai.vision import *


# In[4]:


print(torch.cuda.is_available(), torch.backends.cudnn.enabled)


# In[20]:


path_model='/kaggle/working/'
path_input="/kaggle/input/"
label_df = pd.read_csv(f"{path_input}train.csv")
label_df.head(10)


# In[21]:


label_df.shape


# In[29]:


data_folder = Path("../input")


# In[30]:


data_folder.ls()


# In[31]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/sample_submission.csv")


# In[32]:


test_img = ImageList.from_df(test_df, path=data_folder/'test', folder='test')
trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
train_img = (ImageList.from_df(train_df, path=data_folder/'train', folder='train')
        .split_by_rand_pct(0.01)
        .label_from_df()
        .add_test(test_img)
        .transform(trfm, size=128)
        .databunch(path='.', bs=64, device= torch.device('cuda:0'))
        .normalize(imagenet_stats)
       )


# In[34]:


train_img.show_batch(rows=3, figsize=(8,8))


# In[ ]:





# In[36]:


learner = cnn_learner(train_img,models.resnet50,metrics=[accuracy],model_dir=f'{path_model}')


# In[38]:


lr = 3.5e-02
learner.fit_one_cycle(5, slice(lr))


# In[40]:


preds,_ = learner.get_preds(ds_type=DatasetType.Test)


# In[50]:


test_df.has_cactus = abs(preds.numpy()[:, 0])


# In[51]:


test_df.to_csv('submission.csv', index=False)


# In[52]:


test_df.head()


# In[ ]:




