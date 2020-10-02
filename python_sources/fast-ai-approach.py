#!/usr/bin/env python
# coding: utf-8

# # Set up

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[ ]:


# make sure CUDA is available and enabled
print(torch.cuda.is_available(), torch.backends.cudnn.enabled)


# In[ ]:


comp_name = "digit_recognizer"
input_path = "../input/"
wd = "/kaggle/working/"


# In[ ]:


def create_symlnk(src_dir, lnk_name, dst_dir=wd, target_is_dir=False):
    """
    If symbolic link does not already exist, create it by pointing dst_dir/lnk_name to src_dir/lnk_name
    """
    if not os.path.exists(dst_dir + lnk_name):
        os.symlink(src=src_dir + lnk_name, dst = dst_dir + lnk_name, target_is_directory=target_is_dir)


# In[ ]:


create_symlnk(input_path, "train.csv")
create_symlnk(input_path, "test.csv")


# In[ ]:


# perform sanity check
get_ipython().system('ls -alh')


# In[ ]:


# load data
train_df = pd.read_csv(f"{wd}train.csv")
test_df = pd.read_csv(f"{wd}test.csv")


# In[ ]:


train_df.head()


# In[ ]:


print(train_df.shape, test_df.shape)


# # Prepare Data

# In[ ]:


# create validation dataset
val_df = train_df.sample(frac=0.2, random_state=1337)
val_df.shape


# In[ ]:


# remove validation data from train dataset
train_df = train_df.drop(val_df.index)
train_df.shape


# In[ ]:


# separate labels from data
Y_train = train_df["label"]
Y_valid = val_df["label"]
X_train = train_df.drop("label", axis=1)
X_valid = val_df.drop("label", axis=1)


# In[ ]:


print(X_train.shape, X_valid.shape)
print(Y_train.shape, Y_valid.shape)


# In[ ]:


# display an actual image/digit
img = X_train.iloc[0,:].values.reshape(28,28)
plt.imshow(img, cmap="gray")


# In[ ]:


def reshape_img(matrix):
    """
    Reshape an existing 2D pandas.dataframe into 3D-numpy.ndarray
    """
    try:
        return matrix.values.reshape(-1, 28, 28)
    except AttributeError as e:
        print(e)
        
def add_color_channel(matrix):
    """
    Add missing color channels to previously reshaped image
    """
    matrix = np.stack((matrix, ) *3, axis = -1)
    return matrix

def convert_ndarry(matrix):
    """
    Convert pandas.series into numpy.ndarray
    """
    try:
        return matrix.values.flatten()
    except AttributeError as e:
        print(e)
        
# reshape data and add color channels
X_train = reshape_img(X_train)
X_train = add_color_channel(X_train)
X_valid = reshape_img(X_valid)
X_valid = add_color_channel(X_valid)
test_df = reshape_img(test_df)
test_df = add_color_channel(test_df)

# convert y_train and y_valid into proper numpy.ndarray
Y_train = convert_ndarry(Y_train)
Y_valid = convert_ndarry(Y_valid)


# In[ ]:


# run sanity checks
preprocessed_data = [X_train, Y_train, X_valid, Y_valid, test_df]
print([e.shape for e in preprocessed_data])
print([type(e) for e in preprocessed_data])


# In[ ]:


# define architecture
# arch = resnet50
# sz = 28
# classes = np.unique(Y_train)


# In[ ]:


# data = ImageClassifierData.from_arrays(path=wd, 
#                                        trn=(X_train, Y_train),
#                                        val=(X_valid, Y_valid),
#                                        classes=Y_train,
#                                        test=test_df
#                                        #, tfms=tfms_from_model(arch, sz)
#                                       )


# In[ ]:


# run learner with precompute enabled
# learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[ ]:


# fit learner
# %time learn.fit(0.02, 2)


# ## Find the best Learning Rate

# In[ ]:


# data.classes


# In[ ]:


# lr_finder = learn.lr_find(start_lr=1e-5)


# In[ ]:


# Plotting learning rate across minibatches
# learn.sched.plot_lr()


# In[ ]:


# learn.sched.plot()


# # Image transformations

# In[ ]:


# tfms = tfms_from_model(resnet34, sz, max_zoom=1.1) #aug_tfms=transforms_side_on,

# data = ImageClassifierData.from_arrays(path=wd, 
#                                        trn=(X_train, Y_train),
#                                        val=(X_valid, Y_valid),
#                                        classes=Y_train,
#                                        test=test_df
#                                        , tfms=tfms_from_model(arch, sz)
#                                       )
# learn = ConvLearner.pretrained(arch, data, precompute=True)
# %time learn.fit(0.02, 2)


# # Improving model further

# In[ ]:


# shutil.rmtree(f'{wd}tmp', ignore_errors=True)


# In[ ]:


# Allos us to unfreeze layers
# learn = ConvLearner.pretrained(arch, data, precompute=False)
# %time learn.fit(0.02, 2)


# In[ ]:


# Cycle length - Number of epochs before resseting the learning lenght
# 3 in here is number of epochs
# learn.fit(1e-2, 3, cycle_len=2)


# In[ ]:


# learn.sched.plot_lr()


# In[ ]:


# learn.save('mymodel_lastlayer')


# In[ ]:


# learn.load('mymodel_lastlayer')


# # Unfreezing other layers

# In[ ]:


# shutil.rmtree(f'{wd}tmp', ignore_errors=True)


# In[ ]:


# learn = ConvLearner.pretrained(arch, data, precompute=False)
# learn.unfreeze()


# In[ ]:


# lr=np.array([1e-4,1e-3,1e-2])
# learn.fit(lr, 3, cycle_len=2)


# In[ ]:


# learn.sched.plot_lr()


# # Cycle multiplier

# In[ ]:


arch = resnet50
sz = 28
classes = np.unique(Y_train)

tfms = tfms_from_model(arch, sz, max_zoom=1.1) #aug_tfms=transforms_side_on,

data = ImageClassifierData.from_arrays(path=wd, 
                                       trn=(X_train, Y_train),
                                       val=(X_valid, Y_valid),
                                       classes=classes,
                                       test=test_df,
                                       tfms=tfms_from_model(arch, sz)
                                      )


learn = ConvLearner.pretrained(arch, data, precompute=False)
learn.unfreeze()
lr = np.array([0.001, 0.0075, 0.01])


# In[ ]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[ ]:


learn.save("mymodel_992")


# # Submit results

# In[ ]:


log_preds, y = learn.TTA(is_test=True)


# In[ ]:


probs_test = np.mean(np.exp(log_preds), 0)
results = pd.DataFrame(probs_test)
results.index += 1
results = results.assign(Label = results.values.argmax(axis=1))
results = results.assign(ImageId = results.index.values)
results = results.drop([0,1,2,3,4,5,6,7,8,9], axis=1)


# In[ ]:


results.to_csv(f"submission.csv", index=False)


# In[ ]:




