#!/usr/bin/env python
# coding: utf-8

# In this kernel, we will try to first create a pandas dataframe from the given dataset present as image files and then use it to feed into the training model.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.


# In[ ]:


import glob
train_arr = []
for file in glob.glob("../input/train/train/*/*"):
    train_arr.append({"name": file, "label": file.split("/")[-2]})
df = pd.DataFrame(train_arr)


# In[ ]:


test_df = pd.read_csv(f"../input/sample_submission.csv")


# In[ ]:


df.sample(frac=1).head()


# **Let's inspect the distribution of classes**

# In[ ]:


df["label"].nunique()


# In[ ]:


df["label"].value_counts()


# We can see that there is imbalance in the distribution.

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
import plotly.figure_factory as ff
init_notebook_mode(connected=True)
import matplotlib.cm as cm
import re


# Same could be visualized using plot.ly chart

# In[ ]:


iplot([go.Bar(
x=df["label"].value_counts().keys(),
y=df["label"].value_counts())])


# Let's also see the width/height and aspect ratio for the images.

# In[ ]:


get_ipython().system('pip install imagesize')


# In[ ]:


import imagesize
df["width"] = 0
df["height"] = 0
df["aspect_ratio"] = 0.0
for idx, row in df.iterrows():
    width, height = imagesize.get(row["name"])
    df.at[idx, "width"] = width
    df.at[idx, "height"] = height
    df.at[idx, "aspect_ratio"] = float(height) / float(width)


# In[ ]:


df.head()


# In[ ]:


df["height"].hist()


# In[ ]:


df["width"].hist()


# In[ ]:


df["aspect_ratio"].hist()


# Since all the images are not of same aspect ratio, it will be important to decide how we crop/scale the images while applying augmentations

# In[ ]:


from fastai.vision import *


# <h3> 
# Approach of data loading #1 ImageDataBunch.from_folder<h3>

# In[ ]:


path = Path("../input")
tfms = get_transforms(do_flip=True, max_rotate=10, max_zoom=1.2, max_lighting=0.3, max_warp=0.15)
data = ImageDataBunch.from_folder(path/"train",valid_pct=0.3, ds_tfms=tfms, size=224)


# In[ ]:


data.show_batch(rows=2, figsize=(5,5))


# <h3> 
# Approach of data loading #2 ImageDataBunch.from_df<h3>

# In[ ]:


df.head()


# In[ ]:


data = ImageDataBunch.from_df("", df=df[["name", "label"]], label_col="label", folder="", size=64)


# In[ ]:


data.add_test(ImageList.from_df(test_df, '../input', folder="test/test"))


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(3, 1e-2)


# In[ ]:


learn.lr_find();


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, 1e-25)


# <h3> Using fastai's ClassificationInterpretation to analyze the training results </h3>

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(25,25))


# 

# In[ ]:


interp.plot_top_losses(9, figsize=(25,25))


# In[ ]:


interp.most_confused()


# In[ ]:


test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)
test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]


# In[ ]:


test_df.predicted_class = test_preds
test_df.to_csv("submission.csv", index=False)


# In[ ]:




