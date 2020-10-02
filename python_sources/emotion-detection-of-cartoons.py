#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai.vision import *
from fastai.metrics import accuracy


# In[ ]:


import pandas as pd
train = pd.read_csv("../input/Datasets/Train.csv")


# In[ ]:


train.head()


# In[ ]:


train['Emotion'].value_counts()


# # Image Augmentation Part

# In[ ]:


tfms = get_transforms(do_flip = True, flip_vert = False, max_rotate=10.0)


# In[ ]:


path = "../input/Datasets"


# In[ ]:


np.random.seed(0)
src = (ImageList.from_csv(path, 'Train.csv', folder='Train_Images')
      .split_by_rand_pct(0.2)
      .label_from_df())


# **Resizing Images (128*128) Batch Size = 32**

# In[ ]:


data = (src.transform(tfms, size=128).databunch(bs=32).normalize(imagenet_stats))


# In[ ]:


data


# In[ ]:


data.show_batch(rows=3, figsize=(7,11))


# In[ ]:


arch = models.resnet152


# In[ ]:


from keras.utils.generic_utils import get_custom_objects
from keras import applications
from keras.layers import Activation
import keras
import tensorflow as tf


# **Defining Swish Activation Function**

# In[ ]:


def swish(x,beta = 1):
    return x*sigmoid(beta*x)

get_custom_objects().update({'swish':Activation(swish)})


# In[ ]:


#opt_func:Callable='Adam'
learn = cnn_learner(data, arch, metrics=accuracy)
learn.model_dir='/kaggle/working/'


# In[ ]:


learn.lr_find(num_it=100)
learn.recorder.plot(suggestion=True)


# In[ ]:


#Fitting One Cycle Method in this Problem
lr = 1e-03
learn.fit_one_cycle(10,max_lr=lr)


# In[ ]:


learn.save('cartoon_emotions-1')


# In[ ]:


#Model Interpretation
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(9,figsize=(15,15))


# In[ ]:


test = ImageList.from_folder('../input/Datasets/Test_Images')
len(test)


# In[ ]:


test


# In[ ]:


learn.export(file = Path("/kaggle/working/export.pkl"))


# In[ ]:


deployed_path = "/kaggle/working/"


# In[ ]:


learn = load_learner(deployed_path, test = test)


# In[ ]:


preds, ids = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


preds


# In[ ]:


learn.data.classes[:]


# In[ ]:


learn.data.classes[:]
len(preds)
preds[:10]


# In[ ]:


df = pd.DataFrame(preds, columns=learn.data.classes)
df.index+=1
df = df.assign(Label = df.values.argmax(axis=1))


# In[ ]:


df.head(10)


# In[ ]:


df = df.assign(image =  "fnames")
df = df.replace({'Label':{0:'Unknown', 1:'angry', 2:'happy', 3:'sad', 4:'surprised'}})
df = df.drop(['angry','happy','Unknown','sad','surprised'], axis=1)
df[:10]


# In[ ]:


df = df[['image', 'Label']]
df = df.rename({'image': 'Image'}, axis=1)
thresh = 0.30
labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
labelled_preds[:5]


# In[ ]:


fnames = [f.name[:-4] for f in learn.data.test_ds.items]

fnames[:5]

suffix='.jpg'

fnames= [sub+suffix for sub in fnames]

learn.data.test_ds.items[:5]

df = pd.DataFrame({'Frame_ID':fnames, 'Emotion':labelled_preds})

df.head()


# In[ ]:


df.to_csv('submission.csv',index=False)


# In[ ]:


df.isnull().sum()


# In[ ]:




