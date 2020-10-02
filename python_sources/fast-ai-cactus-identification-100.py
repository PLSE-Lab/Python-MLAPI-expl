#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')
import seaborn as sns
from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('precision', '2')


# In[ ]:


from pathlib import Path
from fastai import *
from fastai.vision import *
import torch


# In[ ]:


data_folder = Path("../input/")
data_folder.ls()


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


def display_all(df):
    with pd.option_context("max_rows",100,"max_columns",df.shape[0]):
        display(df)


# In[ ]:


display_all(train_df)


# In[ ]:


test_img = ImageList.from_df(test_df,path=data_folder/'test',folder='test')
trfm = get_transforms(do_flip=True,flip_vert=True,max_rotate=10.0,max_zoom=1.1,                       max_lighting=0.2,max_warp=0.2,p_affine=0.75,p_lighting=0.75)


# In[ ]:


train_img = (ImageList.from_df(train_df, path=data_folder/'train', folder='train')
        .split_by_rand_pct(0.01)
        .label_from_df()
        .add_test(test_img)
        .transform(trfm, size=128)
        .databunch(path='.', bs=64, device= torch.device('cuda:0'))
        .normalize(imagenet_stats)
       )


# In[ ]:


train_img.show_batch(rows=5, figsize=(12,12))


# In[ ]:


learn = cnn_learner(train_img, models.densenet161, metrics=[error_rate, accuracy])


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 3e-02
learn.fit_one_cycle(5,slice(lr))


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(1, slice(1e-06))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(7,6))


# In[ ]:


preds,_ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


test_df.has_cactus = preds.numpy()[:, 0]


# In[ ]:


test_df.to_csv('submission.csv', index=False)


# In[ ]:



