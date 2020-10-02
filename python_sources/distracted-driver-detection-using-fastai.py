#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.vision import *
from fastai.metrics import error_rate

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


os.listdir("../input/")


# In[ ]:


base_path = "../input/" + os.listdir("../input")[0] + "/"
os.listdir(base_path)


# In[ ]:


drivers_df = pd.read_csv(base_path+"driver_imgs_list.csv")


# In[ ]:


drivers_df.head()


# In[ ]:


categories = {
"c0": "safe driving",
"c1": "texting - right",
"c2": 'talking on the phone - right',
"c3": "texting - left",
"c4": "talking on the phone - left",
'c5': "operating the radio",
'c6': 'drinking',
'c7': 'reaching behind',
'c8': 'hair and makeup',
'c9': 'talking to passenger'
}


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'ImageDataBunch.from_folder')


# In[ ]:


imgs_path = base_path + "imgs/"
data = ImageDataBunch.from_folder(imgs_path, train=imgs_path+"train", valid_pct=0.2, test=imgs_path+"test",
                                    ds_tfms=get_transforms(), size=224, bs=16).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=5, figsize=(8,10))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.model_dir='/kaggle/working/'


# In[ ]:


learn.save("stage-1")


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


confused = interp.most_confused(min_val=2)


# In[ ]:


for x in confused:
    print("Real:",categories[x[0]],", Predicted:", categories[x[1]],", Number of times it did it:", x[2])


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-4))


# In[ ]:


learn.save("stage-2")


# In[ ]:


get_ipython().system('pip install pytorch2keras')


# # Trying to convert Pytorch(fastai) model to keras/tf

# In[ ]:


get_ipython().system('pip install onnx')


# In[ ]:


pytorch_model = learn.model_dir+"stage-2.pth"
keras_output = learn.model_dir+"learn.h5"


# In[ ]:


import tensorflow as tf
import torch
import onnx


# In[ ]:


# To Be Continued


# In[ ]:




