#!/usr/bin/env python
# coding: utf-8

# ### Thanks to authors of these notebooks for helpful examples of how to imitate the fast.ai MOOC code within a kernel
# 
# https://www.kaggle.com/stefanbuenten/dog-breed-test-with-fastai
# 
# https://www.kaggle.com/hortonhearsafoo/fast-ai-lesson-1

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import numpy as np
import pandas as pd
import os


# In[ ]:


labels = pd.read_csv("../input/labels.csv")


# In[ ]:


## validation data
#val_idxs = get_cv_idxs(labels.shape[0])


# In[ ]:


PATH = "../input/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"


# In[ ]:


arch=resnet101
sz=224
bs=48


# In[ ]:


tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)


# In[ ]:


data = ImageClassifierData.from_csv(path=PATH, folder="train", csv_fname=f"{PATH}labels.csv", tfms=tfms, suffix=".jpg", test_name="test",bs=bs,num_workers=4)


# In[ ]:


learn = ConvLearner.pretrained(arch, data, precompute=True, tmp_name=TMP_PATH, models_name=MODEL_PATH, ps=0.4)


# In[ ]:


lrf = learn.lr_find()
learn.sched.plot()


# In[ ]:


def get_data(sz,bs):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom = 1.1)
    data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', test_name='test', suffix='.jpg',tfms=tfms, bs=bs, num_workers=4)
    return data if sz > 300 else data.resize(sz, '/tmp')


# In[ ]:


##without augmentation
learn.fit(1e-1, 5)


# In[ ]:


##with augmentation
learn.precompute=False
learn.fit(1e-1, 5, cycle_len=1)


# In[ ]:


## increase size trick
learn.set_data(get_data(299,bs))


# In[ ]:


##and now resized
learn.fit(1e-1, 3, cycle_len=1)


# In[ ]:


learn.fit(1e-1,3,cycle_len=1,cycle_mult=2)


# In[ ]:


learn.fit(1e-1, 1, cycle_len=2)


# In[ ]:


## unfreezing doesnt help in this case....


# In[ ]:


from sklearn.metrics import log_loss
log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds), 0)
accuracy_np(probs, y), log_loss(y, probs)


# In[ ]:


log_preds_test, y_test = learn.TTA(is_test=True)
probs_test = np.mean(np.exp(log_preds_test), 0)
df = pd.DataFrame(probs_test)
df.columns = data.classes
df.insert(0, "id", [e[5:-4] for e in data.test_ds.fnames])
df.to_csv("submission.csv", index=False)

