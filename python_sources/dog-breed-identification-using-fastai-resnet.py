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


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[ ]:


torch.cuda.set_device(0)


# In[ ]:


torch.backends.cudnn.enabled


# In[ ]:


PATH = '../input/'
sz = 224
arch = resnet101
bs = 128


# In[ ]:


get_ipython().system('ln -s {PATH}train')
get_ipython().system('ln -s {PATH}test')
get_ipython().system('ls')


# In[ ]:


label_csv = f'{PATH}labels.csv'
n = len(list(open(label_csv))) - 1 # header is not counted (-1)
val_idxs = get_cv_idxs(n) # random 20% data for validation set


# In[ ]:


def get_data(sz,bs):
    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_csv('.', 'train', label_csv, test_name='test',
                                       val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)
    return data if sz > 300 else data.resize(340, 'tmp')


# In[ ]:


data = get_data(sz,bs)


# In[ ]:


fn = f'{PATH}' + data.trn_ds.fnames[0]; fn
img = PIL.Image.open(fn); img


# In[ ]:


learn = ConvLearner.pretrained(arch,data,precompute=True)


# In[ ]:


learning_rate = learn.lr_find()
learn.sched.plot()


# In[ ]:


learn.fit(1e-2, 5)


# In[ ]:


from sklearn import metrics


# In[ ]:


#submitting...
log_preds, y = learn.TTA(is_test=True) # use test dataset rather than validation dataset
probs = np.mean(np.exp(log_preds),0)


# In[ ]:


probs.shape


# In[ ]:


df = pd.DataFrame(probs)
df.columns = data.classes


# In[ ]:


df.insert(0, 'id', [o[5:-4] for o in data.test_ds.fnames])
df.head()


# In[ ]:



wd = '/kaggle/working/'
def clean_up(wd=wd):
    """
    Delete all temporary directories and symlinks in working directory (wd)
    """
    for root, dirs, files in os.walk(wd):
        try:
            for d in dirs:
                if os.path.islink(d):
                    os.unlink(d)
                else:
                    shutil.rmtree(d)
            for f in files:
                if os.path.islink(f):
                    os.unlink(f)
                else:
                    print(f)
        except FileNotFoundError as e:
            print(e)


# In[ ]:


#SUBM = f'..input/subm/'
#os.makedirs(SUBM, exist_ok=True)
#df.to_csv(f'{SUBM}subm.gz', compression='gzip', index=False)
df.to_csv('submission.csv', index=False)


# In[ ]:


clean_up()


# In[ ]:




