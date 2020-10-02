#!/usr/bin/env python
# coding: utf-8

# # work in progress

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# thanks to [this kernel](https://www.kaggle.com/xhlulu/recursion-2019-load-resize-and-save-images) and [this dataset](https://www.kaggle.com/xhlulu/recursion-cellular-image-classification-224-jpg)

# In[ ]:


import os
import sys
import zipfile

import pandas as pd
import numpy as np

from fastai import *
from fastai.vision import *

from tqdm import tqdm

from pathlib import Path
from PIL import Image


# In[ ]:


# hide warnings
import warnings
warnings.simplefilter('ignore')


# In[ ]:


os.listdir("../input")


# In[ ]:


input_dir = Path("../input/recursion-cellular-image-classification-224-jpg")
input_dir1 = Path("../input/recursion-cellular-image-classification")
print(os.listdir(input_dir))


# In[ ]:


train_dir = input_dir/"train/train"
test_dir = input_dir/"test/test"
print(os.listdir(train_dir)[:3])
print(os.listdir(test_dir)[:3])


# In[ ]:


train_csv = pd.read_csv(input_dir/"new_train.csv")
train_csv = train_csv[["filename", "sirna"]]


# In[ ]:


test_dataset = ImageList.from_folder(input_dir/"test/test")
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_df(
    path = input_dir,
    df = train_csv,
    folder = "train/train",
    valid_pct = 0.2,
    bs = 32,
    size = 224,
    ds_tfms = tfms,
    num_workers = 0
)
data.add_test(test_dataset)
data.normalize(imagenet_stats)
print(data)
print(len(data.classes))
data.show_batch(rows=3, figsize=(10,10))


# In[ ]:


learn = cnn_learner(data, models.resnet101, metrics=accuracy, model_dir="/tmp/models")


# In[ ]:


learn.fit_one_cycle(5)


# In[ ]:


class_score, y = learn.get_preds(DatasetType.Test)


# In[ ]:


sample_submission = pd.read_csv(input_dir1/"sample_submission.csv")
sample_submission.head()


# In[ ]:


class_score


# In[ ]:


sample_submission.id_code.values[0] in os.listdir(test_dir)


# In[ ]:


all_test_files = [element[:-8] for element in os.listdir(test_dir)]
all_test_files[:5]


# In[ ]:


def get_class_score_both(id_code):
    index1 = all_test_files.index(id_code)
    index2 = all_test_files.index(id_code, index1+1)
    sum_class_score = class_score[index1] + class_score[index2]
    avg_class_score = sum_class_score / 2
    return(avg_class_score)


# In[ ]:


def get_class_score_s1(id_code):
    index1 = all_test_files.index(id_code)
    index2 = all_test_files.index(id_code, index1+1)
    sum_class_score = class_score[index1] + class_score[index2]
    avg_class_score = sum_class_score / 2
    return(avg_class_score)


# In[ ]:


def get_class_score_s2(id_code):
    index1 = all_test_files.index(id_code)
    index2 = all_test_files.index(id_code, index1+1)
    sum_class_score = class_score[index1] + class_score[index2]
    avg_class_score = sum_class_score / 2
    return(avg_class_score)


# In[ ]:


x = list(map(get_class_score_both, sample_submission.id_code.values))
x = torch.stack(x)
x = x.argmax(dim=1)
x


# In[ ]:


submission  = pd.DataFrame({
    "id_code": sample_submission.id_code,
    "sirna": x
})
submission.to_csv("submission.csv", index=False)
submission[:10]

