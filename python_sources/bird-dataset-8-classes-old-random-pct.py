#!/usr/bin/env python
# coding: utf-8

# New data set made up of 8 classes, with similar distribution of images numbers as the whole dataset.

# ### Setup

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import os, os.path

import pandas as pd

from fastai.callbacks.tracker import *


# In[ ]:


batchsize = 64
default_epoch = 50 #might be cutoff by callback
np.random.seed(2)


# ### Data gathering
# 
# We import the dataset samples (images, organised in folders) together with their labels (from CSV). We use the labels to split the dataset into training and test partitions (85/15) by using two separate label files (_train.csv; _test.csv).
# For automatic preparation of the partitions, see `utils/dataset_splitter_toCSV.py`.
# 
# 
# As for a validation set, we will use ImageDataBunch's built-in splitting to separate off the validation partition.

# Get path of the image folders:

# In[ ]:


path = Path('/kaggle/input/birds-uk-8classes-subset/Aves_subset_8classes_distributionSimilarToFullSet')
path.ls()


# Get sample count:

# In[ ]:


# labels_to_images_list = []
# folders = os.listdir(path)
# for folder in folders:
#     folder_path = os.path.join(path, folder)
#     if (os.path.isdir(folder_path)):
#         if (len(os.listdir(folder_path)) is not None):
#             files = os.listdir(folder_path)
#             for file in files:
#                 labels_to_images_list.append([folder,file])
# len_dataset = len(labels_to_images_list)
# len_dataset


# Load the path of the image name csvs

# In[ ]:


csv_train = Path('/kaggle/input/smallset-csvs/Aves_subset_8classes_distributionSimilarToFullSet_train.csv')
csv_test = Path('/kaggle/input/smallset-csvs/Aves_subset_8classes_distributionSimilarToFullSet_test.csv')


# Create image lists of train and test images

# In[ ]:


imagelist_train = ImageList.from_csv(path,csv_train)
imagelist_test = ImageList.from_csv(path,csv_test)


# Create dataframe of image names and labels. Concatenate train and test to make df_all

# In[ ]:


df_train = pd.read_csv(csv_train, header=None)
df_test = pd.read_csv(csv_test, header=None)

df_all = df_train.append(df_test)

df_all


# In[ ]:


len_dataset = len(df_all.index)
len_dataset


# ### Preparations

# Set up data bunch, i.e. 1) load image to label; 2) split dataset into training and validation partition

# In[ ]:


tfms = get_transforms()


# In[ ]:


#create training databunch

data = ImageDataBunch.from_csv(path=path, csv_labels=csv_train, valid_pct=0.2, seed=seed,delimiter=",", ds_tfms=tfms, size=256, bs=batchsize, num_workers=0).normalize(imagenet_stats)

data


# num_workers is the number of CPUs to use

# In[ ]:


data = (ImageList.from_csv(path,csv_train,delimiter=",")
        .split_by_rand_pct(valid_pct=0.2)
        .label_from_df(cols=1) 
        .transform(tfms, size=256)
        .databunch(bs=batchsize, num_workers=0)
        .normalize(imagenet_stats))

data


# In[ ]:


#create test databunch

data_test = (ImageList.from_df(df_all, path)
        .split_by_list(imagelist_train, imagelist_test)
        .label_from_df(cols=1) 
        .transform(tfms, size=256)
        .databunch(bs=batchsize, num_workers=0)
        .normalize(imagenet_stats))

data_test


# **Tweaking watchlist**
# get_transforms: look into how it works and how to adjust (data augmentation) (L6)
# 
# normalize(image_stats): look into

# In[ ]:


print(data.classes)


# In[ ]:


data.items


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# Create learner with metrics and ensure output from learn directs to kaggle working directory

# In[ ]:


kappa = KappaScore()
kappa.weights = "quadratic"

learn = cnn_learner(data, models.resnet34, metrics=[accuracy,kappa],callback_fns=[partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=5, mode = 'auto')])
learn.model_dir = "/kaggle/working"


# ### Frozen Training

# For an explanation of freezing and unfreezing, see fastai L2 and start of L5
# 
# For a summary of how to select LR, see fastai L3

# In[ ]:


#default LR here is 3e-3
learn.fit_one_cycle(4)


# In[ ]:


learn.unfreeze()


# In[ ]:


#learn.lr_find()


# In[ ]:


#learn.recorder.plot()


# ### Unfrozen Training

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(default_epoch, max_lr=slice(3e-6,1e-4)) # <-- ADJUST depending on lr_finder plot outcome


# Plot the losses over epochs

# In[ ]:


learn.recorder.plot_losses()


# ### Analysis of Training

# Confusion matrix and top losses:

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.most_confused(min_val=2)


# ### Test Set

# Switch the learner's dataloader to the test dataloader (allows the use of the same learner for the test analysis)

# In[ ]:


learn.data.valid_dl = data_test.valid_dl
learn.unfreeze()


# In[ ]:


learn.validate(data.valid_dl, metrics = [accuracy,kappa])


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.most_confused(min_val=2)


# ### Model export

# In[ ]:


learn.export("/kaggle/working/model.pkl")

