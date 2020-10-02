#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.imports import *
from fastai.torch_imports import *
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


os.listdir("../working")


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


torch.cuda.set_device(0)


# In[ ]:


comp_name = "dog_breed"
input_path = "../input/"
wd = "/kaggle/working/"


# In[ ]:


def create_symlnk(src_dir, src_name, dst_name, dst_dir=wd, target_is_dir=False):
    """
    If symbolic link does not already exist, create it by pointing dst_dir/lnk_name to src_dir/lnk_name
    """
    if not os.path.exists(dst_dir + dst_name):
        os.symlink(src=src_dir + src_name, dst = dst_dir + src_name, target_is_directory=target_is_dir)


# In[ ]:


PATH = wd
sz = 224
arch = resnext101_64
bs = 58


# In[ ]:


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


create_symlnk(input_path, "train", "train", target_is_dir=True)
create_symlnk(input_path, "test", "test", target_is_dir=True)
create_symlnk(input_path, "labels.csv", "labels.csv")


# In[ ]:


label_df = pd.read_csv(f"{wd}labels.csv")


# In[ ]:


val_idxs = get_cv_idxs(label_df.shape[0])


# In[ ]:


arch = resnet101
sz = 224
bs = 64


# In[ ]:


val_idxs


# In[ ]:


tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_csv(path=wd, folder="train", csv_fname=f"{wd}labels.csv", tfms=tfms, val_idxs=val_idxs, suffix=".jpg", test_name="test")


# In[ ]:


learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[ ]:


learn.fit(1e-2, 3)


# In[ ]:


from sklearn import metrics


# In[ ]:


log_preds, y = learn.TTA(is_test=True) # use test dataset rather than validation dataset
probs = np.mean(np.exp(log_preds),0)
#accuracy_np(probs, y), metrcs.log_loss(y, probs) # This does not make sense since test dataset has no labels


# In[ ]:


df = pd.DataFrame(probs)
df.columns = data.classes


# In[ ]:


df.insert(0, "id", [e[5:-4] for e in data.test_ds.fnames])


# In[ ]:


df.to_csv(f"sub_{comp_name}_{str(arch.__name__)}.csv", index=False)


# In[ ]:


clean_up()


# In[ ]:




