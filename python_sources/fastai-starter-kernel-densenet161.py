#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import numpy as np
import pandas as pd

from fastai.vision import *
from fastai.callbacks import *

from pathlib import Path

import os
import shutil

np.random.seed(42)


# In[ ]:


os.listdir('../input/')


# In[ ]:


data_folder = Path("../input/")
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


train = train_df.copy()
test = test_df.copy()


# In[ ]:


train['id_code'] = train['id_code'].apply(lambda x: str(x) + str(".png"))


# In[ ]:


test['id_code'] = test['id_code'].apply(lambda x: str(x) + str(".png"))


# In[ ]:


test_img = ImageList.from_df(test, path=data_folder, folder='test_images')
trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
train_img = (ImageList.from_df(train, path=data_folder, folder='train_images')
        .split_by_rand_pct(0.01)
        .label_from_df()
        .add_test(test_img)
        .transform(trfm, size=128)
        .databunch(path='.', bs=64, device= torch.device('cuda:0'))
        .normalize(imagenet_stats)
       )


# In[ ]:


learn50 = cnn_learner(train_img, models.densenet161, metrics=[error_rate, accuracy], model_dir="/tmp/model/")


# In[ ]:


lr = 3e-02
learn50.fit_one_cycle(4 , slice(lr))


# In[ ]:


preds,_ = learn50.get_preds(ds_type=DatasetType.Test)


# In[ ]:


test_df.head()


# In[ ]:


test_df.diagnosis = np.argmax(preds,axis=1)
test_df.head()


# In[ ]:


test_df.to_csv('submission.csv', index=False)

