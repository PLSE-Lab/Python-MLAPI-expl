#!/usr/bin/env python
# coding: utf-8

# # About
# This notebook naively applies the techniques from [fastai's deep learning for coders course](http://course.fast.ai/) - specifically those from [lesson 1](http://course.fast.ai/lessons/lesson1.html) - to the MNIST dataset

# # Setup

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


# ## Helper functions to deal with Kaggle's file system limitations

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


# # Inspect data

# In[ ]:


# load data
train_df = pd.read_csv(f"{wd}train.csv")
test_df = pd.read_csv(f"{wd}test.csv")


# In[ ]:


train_df.head()


# In[ ]:


print(train_df.shape, test_df.shape)


# # Prepare data

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


# # Preprocessing
# Preprocessing according to advice in [fastai's forums](http://forums.fast.ai/t/how-to-use-kaggles-mnist-data-with-imageclassifierdata/7653/9)

# In[ ]:


def reshape_img(matrix):
    """
    Reshape an existing 2D pandas.dataframe into 3D-numpy.ndarray
    """
    try:
        return matrix.values.reshape(-1, 28, 28)
    except AttributeError as e:
        print(e)


# In[ ]:


def add_color_channel(matrix):
    """
    Add missing color channels to previously reshaped image
    """
    matrix = np.stack((matrix, ) *3, axis = -1)
    return matrix


# In[ ]:


def convert_ndarry(matrix):
    """
    Convert pandas.series into numpy.ndarray
    """
    try:
        return matrix.values.flatten()
    except AttributeError as e:
        print(e)


# In[ ]:


# reshape data and add color channels
X_train = reshape_img(X_train)
X_train = add_color_channel(X_train)
X_valid = reshape_img(X_valid)
X_valid = add_color_channel(X_valid)
test_df = reshape_img(test_df)
test_df = add_color_channel(test_df)


# In[ ]:


# convert y_train and y_valid into proper numpy.ndarray
Y_train = convert_ndarry(Y_train)
Y_valid = convert_ndarry(Y_valid)


# In[ ]:


# run sanity checks
preprocessed_data = [X_train, Y_train, X_valid, Y_valid, test_df]
print([e.shape for e in preprocessed_data])
print([type(e) for e in preprocessed_data])


# # Model

# In[ ]:


# define architecture
arch = resnet50
sz = 28
classes = np.unique(Y_train)


# In[ ]:


data = ImageClassifierData.from_arrays(path=wd, 
                                       trn=(X_train, Y_train),
                                       val=(X_valid, Y_valid),
                                       classes=Y_train,
                                       test=test_df,
                                       tfms=tfms_from_model(arch, sz))


# In[ ]:


# run learner with precompute enabled
learn = ConvLearner.pretrained(arch, data, precompute=True)


# In[ ]:


# find optimal learning rate
lrf = learn.lr_find()


# In[ ]:


# plot loss vs. learning rate
learn.sched.plot()


# In[ ]:


# fit learner
get_ipython().run_line_magic('time', 'learn.fit(1e-2, 2)')


# In[ ]:


# save model
#learn.save("28_lastlayer")


# In[ ]:


# disable precompute and unfreeze layers
learn.precompute=False
learn.unfreeze()


# In[ ]:


# define differential learning rates
lr = np.array([0.001, 0.0075, 0.01])


# In[ ]:


# retrain full model
get_ipython().run_line_magic('time', 'learn.fit(lr, 3, cycle_len=1, cycle_mult=2)')


# In[ ]:


# save full model
#learn.save("28_all")


# In[ ]:


# get accuracy for validation set
log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds), 0)
accuracy_np(probs, y)


# In[ ]:


# predict on test set
get_ipython().run_line_magic('time', 'log_preds_test, y_test = learn.TTA(is_test=True)')
probs_test = np.mean(np.exp(log_preds_test), 0)
probs_test.shape


# # Submission

# In[ ]:


# create dataframe from probabilities
df = pd.DataFrame(probs_test)


# In[ ]:


# increase index by 1 to obtain proper ImageIDs
df.index += 1


# In[ ]:


# create new colum containing label with highest probability for each digit
df = df.assign(Label = df.values.argmax(axis=1))


# In[ ]:


# replicate index as dedicated ImageID column necessary for submission
df = df.assign(ImageId = df.index.values)


# In[ ]:


# drop individual probabilites
df = df.drop([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], axis=1)


# In[ ]:


# reorder columns for submission
df = df[["ImageId", "Label"]]


# In[ ]:


# run sanity checks
df.head()


# In[ ]:


# ...
df.tail()


# In[ ]:


# ...
df.shape


# In[ ]:


# write dataframe to CSV
df.to_csv(f"sub_{comp_name}_{arch.__name__}.csv", index=False)


# # Cleanup

# In[ ]:


def clean_up():
    """
    Delete all temporary directories and symlinks in the current directory
    """
    try:
        shutil.rmtree("models")
        shutil.rmtree("tmp")
        os.unlink("test.csv")
        os.unlink("train.csv")
    except FileNotFoundError as e:
        print(e)


# In[ ]:


clean_up()


# In[ ]:




