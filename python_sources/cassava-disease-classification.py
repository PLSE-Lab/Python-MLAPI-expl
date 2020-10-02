#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pretrainedmodels')


# In[ ]:


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import fastai
from fastai import vision

import pretrainedmodels as pm


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Create training and validation data
# 
# Code in the cell below is used to split the raw training data set in `../input/train/train` into interim training and 
# validation data sets. The interim training and validation data sets are stored in `./data/interim/train` and 
# `./data/interim/valid`, respectively.
# 
# The raw training data set is divided into 80% interim training data and 20% interim validation data using startified 
# sampling on the image labels so that the distribution of labels in to raw training data is preserved in the interim 
# training and validation data sets.

# In[ ]:


import glob
import os
import shutil
from typing import List

import numpy as np
import pandas as pd
from sklearn import model_selection

PREFIX = "./data"
SEED = 42
TEST_SIZE = 0.2

def _filepaths_to_dataframe(paths: List[str]) -> pd.DataFrame:
    """Converts filepaths to a Pandas DataFrame."""
    results = {"label": [], "filename": []}
    for path in paths:
        _, _, _, _, _label, _ = path.split('/')
        results["label"].append(_label)
        results["filename"].append(path)
    df = (pd.DataFrame
            .from_dict(results))
    return df


def _make_interim_training_data(prefix: str, df: pd.DataFrame) -> None:
    if not os.path.isdir(f"{prefix}/interim/train"):
        os.makedirs(f"{prefix}/interim/train")

    for _, row in df.iterrows():
        label, path = row
        filename = (os.path
                      .basename(path))
        if not os.path.isdir(f"{prefix}/interim/train/{label}"):
            os.mkdir(f"{prefix}/interim/train/{label}")
        shutil.copy(path, f"{prefix}/interim/train/{label}/{filename}")

        
def _make_interim_validation_data(prefix: str, df: pd.DataFrame) -> None:
    if not os.path.isdir(f"{prefix}/interim/valid"):
        os.makedirs(f"{prefix}/interim/valid")

    for _, row in df.iterrows():
        label, path = row
        filename = (os.path
                      .basename(path))
        if not os.path.isdir(f"{prefix}/interim/valid/{label}"):
            os.mkdir(f"{prefix}/interim/valid/{label}")
        shutil.copy(path, f"{prefix}/interim/valid/{label}/{filename}")

        
filepaths = glob.glob(f"../input/train/train/*/*.jpg", recursive=True)
df = _filepaths_to_dataframe(filepaths)
prng = np.random.RandomState(SEED)

training_df, validation_df = model_selection.train_test_split(df,
                                                              test_size=TEST_SIZE,
                                                              random_state=prng,
                                                              stratify=df["label"])
    
if not os.path.isdir(PREFIX):
    os.mkdir(PREFIX)
_make_interim_training_data(PREFIX, training_df)
_make_interim_validation_data(PREFIX, validation_df)



# In[ ]:


fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 8))

_ = (df.loc[:, "label"]
       .value_counts()
       .plot
       .bar(ax=axes[0], title="Raw"))

_ = (training_df.loc[:, "label"]
                .value_counts()
                .plot
                .bar(ax=axes[1], title="Training"))

_ = (validation_df.loc[:, "label"]
                  .value_counts()
                  .plot
                  .bar(ax=axes[2], title="Validation"))


# In[ ]:


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)


# # Creating a `ImageDataBunch`
# 
# To create the training data set we use standard data augmentation techniques. All parameters defining the transformations used for data augmentation are left at their default values (unless otherwise specified).

# In[ ]:


_transform_kwargs = {"do_flip": True,
                     "flip_vert": True,  # default is False
                     "max_rotate": 180,  # default is 10
                     "max_zoom": 1.2,    # default is 1.1
                     "max_lighting": 0.2,
                     "max_warp": 0.2,
                     "p_affine": 0.75,
                     "p_lighting": 0.7,
                    }
        
_transforms = vision.get_transforms(**_transform_kwargs)

_data_bunch_kwargs = {"path": "./data/interim",
                      "train": "train",
                      "valid": "valid",
                      "bs": 16,
                      "size": 448,
                      "ds_tfms": _transforms,
                      "test": "../../../input/test/test",  ## hack to access the test data without copying to ./data
                     }

image_data_bunch = (vision.ImageDataBunch
                          .from_folder(**_data_bunch_kwargs)
                          .normalize())


# In[ ]:


image_data_bunch.train_ds


# In[ ]:


image_data_bunch.valid_ds


# In[ ]:


image_data_bunch.test_ds


# # Exploring the data
# 
# Always important to understand what the images that are being fed into your model actually look like.
# 

# In[ ]:


image_data_bunch.show_batch(figsize=(20,20))


# # Fitting the model

# ## Transfer Learning
# 
# For computer vision applications always start by trying transfer learning with a standard architecture: [SE-ResNeXt-101](https://arxiv.org/pdf/1803.09820.pdf).

# In[ ]:


_base_arch = lambda arg: pm.se_resnext101_32x4d(num_classes=1000, pretrained="imagenet")
learner = vision.cnn_learner(image_data_bunch,
                             base_arch=_base_arch,
                             pretrained=True,
                             metrics=vision.error_rate,
                             model_dir="/kaggle/working/models/se-resnext101-32x4d")


# In[ ]:


learner.summary()


# In[ ]:


learner.lr_find()


# In[ ]:


(learner.recorder
        .plot())


# In[ ]:


def find_optimal_lr(recorder):
    """Extract the optimal learning rate from recorder data."""
    optimal_lr = 0
    minimum_loss = float("inf")
    for loss, lr in zip(recorder.losses, recorder.lrs):
        if loss < minimum_loss:
            optimal_lr = lr
            minimum_loss = loss
    return optimal_lr, minimum_loss


# In[ ]:


# define a callback that stores state of "best" model.
# N.B. best model is re-loaded when training completes
_save_model_kwargs = {"every": "improvement",
                      "monitor": "valid_loss",
                      "name": "best-model-stage-1"}
_save_model = (fastai.callbacks
                     .SaveModelCallback(learner, **_save_model_kwargs))

# if validation loss < training loss either learning rate too low or not enough training epoch
learner.fit_one_cycle(15, callbacks=[_save_model])


# # Exploring the model's predictions

# In[ ]:


clf_interp = (vision.ClassificationInterpretation
                    .from_learner(learner))


# In[ ]:


clf_interp.plot_top_losses(16, figsize=(20,20))


# In[ ]:


clf_interp.plot_confusion_matrix()


# In[ ]:


clf_interp.most_confused()


# ## Unfreezing, fine-tuning, and learning rates

# In[ ]:


learner.unfreeze()


# In[ ]:


learner.summary()


# In[ ]:


learner.lr_find()


# In[ ]:


(learner.recorder
        .plot())


# In[ ]:


_save_model_kwargs = {"every": "improvement",
                      "monitor": "valid_loss",
                      "name": "best-model-stage-2"}
_save_model = (fastai.callbacks
                     .SaveModelCallback(learner, **_save_model_kwargs))
learner.fit_one_cycle(15, max_lr=slice(1e-6, 1e-4), callbacks=[_save_model])


# ## Exploring results of the fine-tuned model

# In[ ]:


clf_interp = (vision.ClassificationInterpretation
                    .from_learner(learner))


# In[ ]:


clf_interp.plot_confusion_matrix()


# ## Test-Time Augmentation (TTA)

# In[ ]:


predicted_class_probabilities, _ = learner.TTA(ds_type=fastai.basic_data.DatasetType.Test)


# ## Creating a submission

# In[ ]:


_predicted_classes = (predicted_class_probabilities.argmax(dim=1)
                                                   .numpy())
_class_labels = np.array(['cbb','cbsd','cgm','cmd','healthy'])
_predicted_class_labels = _class_labels[_predicted_classes]

_filenames = np.array([item.name for item in image_data_bunch.test_ds.items])

submission = (pd.DataFrame
                .from_dict({'Category': _predicted_class_labels,'Id': _filenames}))


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv', header=True, index=False)


# In[ ]:


shutil.rmtree(PREFIX)  # necessary not to overwhlem Kaggle with unused output files

