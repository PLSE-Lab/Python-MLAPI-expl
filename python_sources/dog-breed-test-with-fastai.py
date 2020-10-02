#!/usr/bin/env python
# coding: utf-8

# # About
# This kernel applies the techniques from [fastai's deep learning for coders](http://course.fast.ai) course to the dogbreed dataset
# 
# The resulting Kaggle score is 0.34460 which translates to roughly 531th position on the leaderboard.

# # Setup

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd
import os

from fastai.conv_learner import *


# In[ ]:


# make sure CUDA is available and enabled
print(torch.cuda.is_available(), torch.backends.cudnn.enabled)


# In[ ]:


comp_name = "dog_breed"
input_path = "../input/"
wd = "/kaggle/working/"


# ## Helper functions to deal with Kaggle's file system limitations

# In[ ]:


def create_symlnk(src_dir, src_name, dst_name, dst_dir=wd, target_is_dir=False):
    """
    If symbolic link does not already exist, create it by pointing dst_dir/lnk_name to src_dir/lnk_name
    """
    if not os.path.exists(dst_dir + dst_name):
        os.symlink(src=src_dir + src_name, dst = dst_dir + dst_name, target_is_directory=target_is_dir)


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


# perform sanity check
get_ipython().system('ls -alh')


# # Exploration

# In[ ]:


label_df = pd.read_csv(f"{wd}labels.csv")


# In[ ]:


label_df.head()


# In[ ]:


label_df.shape


# In[ ]:


label_df.pivot_table(index="breed", aggfunc=len).sort_values("id", ascending=False)


# In[ ]:


# create validation dataset
val_idxs = get_cv_idxs(label_df.shape[0])


# In[ ]:


# define architecture
arch = resnet34
sz = 224
bs = 64


# In[ ]:


# load data
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_csv(path=wd, folder="train", csv_fname=f"{wd}labels.csv", tfms=tfms, val_idxs=val_idxs, suffix=".jpg", test_name="test")


# In[ ]:


[print(len(e)) for e in [data.trn_ds, data.val_ds, data.test_ds]]


# In[ ]:


# look at an actual image
fn = wd + data.trn_ds.fnames[-1]
img = PIL.Image.open(fn); img


# In[ ]:


img.size


# # Preprocess data

# In[ ]:


def get_data(sz=sz, bs=bs, data=data):
    """
    Load images via fastai's ImageClassifierData.from_csv() object defined as 'data' before
    Return images if size bigger than 300 pixels, else resize to 340 pixels
    """
    return data if sz > 300 else data.resize(340, new_path=wd)


# In[ ]:


data = get_data()


# # Model

# ## Baseline

# In[ ]:


learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[ ]:


lrf = learn.lr_find()


# In[ ]:


learn.sched.plot()


# In[ ]:


# fit baseline model without data augmentation
learn.fit(1e-1, 2)


# In[ ]:


# disable precompute and fit model with data augmentation
learn.precompute=False
learn.fit(1e-1, 5, cycle_len=1)


# In[ ]:


#learn.save("224_pre")


# In[ ]:


#learn.load("224_pre")


# ## Model with increased image size

# In[ ]:


learn.set_data(get_data(299, bs))


# In[ ]:


learn.fit(1e-1, 3, cycle_len=1)


# In[ ]:


from sklearn.metrics import log_loss

log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds), 0)
accuracy_np(probs, y), log_loss(y, probs)


# In[ ]:


#learn.save("299_pre")


# In[ ]:


#learn.load("299_pre")


# ## Prediction on test set

# In[ ]:


log_preds_test, y_test = learn.TTA(is_test=True)
probs_test = np.mean(np.exp(log_preds_test), 0)


# # Submission

# In[ ]:


df = pd.DataFrame(probs_test)
df.columns = data.classes


# In[ ]:


# insert clean ids - without folder prefix and .jpg suffix - of images as first column
df.insert(0, "id", [e[5:-4] for e in data.test_ds.fnames])


# In[ ]:


df.to_csv(f"sub_{comp_name}_{str(arch.__name__)}.csv", index=False)


# In[ ]:


clean_up()

