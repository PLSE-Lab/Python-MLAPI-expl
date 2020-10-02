#!/usr/bin/env python
# coding: utf-8

# # Aerial Cactus Identification

# ## Initialize library

# In[ ]:


import numpy as np
import pandas as pd

from fastai.vision import *
from fastai.callbacks import *

from pathlib import Path

import os
import shutil

np.random.seed(10)

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


data_folder = Path("../input")
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


test_img = ImageList.from_df(test_df, path=data_folder/'test', folder='test')

trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=12.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)

train_img = (ImageList.from_df(train_df, path=data_folder/'train', folder='train')
        .split_by_rand_pct(0.01)
        .label_from_df()
        .add_test(test_img)
        .transform(trfm, size=128)
        .databunch(path='.', bs=64, device= torch.device('cuda:0'))
        .normalize(imagenet_stats)
       )


# ## Training DenseNet-201

# In[ ]:


learn_densnet = cnn_learner(train_img, models.densenet201, metrics=[error_rate, accuracy], model_dir="/tmp/model/")


# In[ ]:


learn_densnet.lr_find()
learn_densnet.recorder.plot()


# In[ ]:


lr = 1e-02
learn_densnet.fit_one_cycle(5 , slice(lr))


# In[ ]:


preds,_ = learn_densnet.get_preds(ds_type=DatasetType.Test)


# In[ ]:


test_df.has_cactus = preds.numpy()[:, 0]
test_df.head()


# In[ ]:


test_df.to_csv('submission.csv', index=False)

