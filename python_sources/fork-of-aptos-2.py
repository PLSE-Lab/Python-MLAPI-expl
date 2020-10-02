#!/usr/bin/env python
# coding: utf-8

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


import numpy as np
import cv2


# In[ ]:


os.mkdir("../train_images")
os.mkdir("../test_images")


# In[ ]:


# for x in os.listdir("../input/aptos2019-blindness-detection/train_images/"):

#     img = cv2.imread("../input/aptos2019-blindness-detection/train_images/"+x,0)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     cl1 = clahe.apply(img)
#     cv2.imwrite("../train_images/"+x,cl1)
    
# for x in os.listdir("../input/aptos2019-blindness-detection/test_images/"):

#     img = cv2.imread("../input/aptos2019-blindness-detection/test_images/"+x,0)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     cl1 = clahe.apply(img)
#     cv2.imwrite("../test_images/"+x,cl1)

for x in os.listdir("../input/aptos2019-blindness-detection/train_images/"):
    img = cv2.imread("../input/aptos2019-blindness-detection/train_images/"+x,0)
    equ = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(equ)
    cv2.imwrite("../train_images/"+x,cl1)
    
for x in os.listdir("../input/aptos2019-blindness-detection/test_images/"):
    img = cv2.imread("../input/aptos2019-blindness-detection/test_images/"+x,0)
    equ = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(equ)
    cv2.imwrite("../test_images/"+x,cl1)


# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.metrics import accuracy, error_rate


# In[ ]:


test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_img = ImageList.from_df(test_df, path="..", folder='/test_images',suffix='.png')


# In[ ]:


train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
tfms = get_transforms(do_flip=True)


# In[ ]:


np.random.seed(5)
data = (ImageList.from_df(train_df,path="..",folder="/train_images",suffix='.png')
        .split_by_rand_pct()
        .label_from_df(cols='diagnosis')
        .add_test(test_img)
        .transform(tfms,size = 256)
        .databunch(bs=64)    
        .normalize(imagenet_stats)
       )


# In[ ]:


data.show_batch(rows=3,figsize = (5,5))


# In[ ]:


data.valid_ds.classes


# In[ ]:


Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
get_ipython().system('cp ../input/resnet50/resnet50.pth /tmp/.cache/torch/checkpoints/resnet50.pth')


# In[ ]:


get_ipython().system('mv  /tmp/.cache/torch/checkpoints/resnet50.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth')


# In[ ]:


model = cnn_learner(data,models.resnet50, metrics = [accuracy,error_rate],callback_fns=ShowGraph)


# In[ ]:


model.summary()


# In[ ]:


model.model_dir = '../kaggle/working/models/'


# In[ ]:


# model.lr_find()
# model.recorder.plot(suggestion = True)


# In[ ]:


model.fit_one_cycle(5,1e-2)


# In[ ]:


model.unfreeze()
# model.lr_find()
# model.recorder.plot(suggestion = True)


# In[ ]:


model.fit_one_cycle(10,max_lr = slice(1e-6,1e-4))


# In[ ]:


model.recorder.plot_losses()


# In[ ]:


interpreter = ClassificationInterpretation.from_learner(model)
interpreter.plot_confusion_matrix()


# In[ ]:


preds, _ = model.get_preds(ds_type=DatasetType.Test)


# In[ ]:


test_df["diagnosis"] = preds.argmax(1)


# In[ ]:


test_df.to_csv('submission.csv', index=False)

