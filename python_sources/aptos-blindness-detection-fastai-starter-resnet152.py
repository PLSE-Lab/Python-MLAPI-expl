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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai import *
import torch
from fastai.metrics import KappaScore
from fastai.vision import *
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.callbacks.hooks import *


# In[ ]:


# copy pretrained weights for resnet152 to the folder fastai will search by default
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
get_ipython().system("cp '../input/resnet152/resnet152.pth' '/tmp/.cache/torch/checkpoints/resnet152-b121ed2d.pth'")


# In[ ]:


train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_df.head()


# In[ ]:


test_df = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
# test_df.id_code = test_df.id_code + '.'
test_img = ImageList.from_df(test_df, path="../input/aptos2019-blindness-detection", folder='/test_images',suffix='.png')
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)


# In[ ]:


np.random.seed(145)
data = (ImageList.from_df(train_df,path="../input/aptos2019-blindness-detection",folder="/train_images",suffix='.png')
        .split_by_rand_pct()
#         .split_none()
        .label_from_df()
        .add_test(test_img)
        .transform(tfms,size = 200)
        .databunch(path='.',bs=16)    
        .normalize(imagenet_stats)
       )


# In[ ]:


data.show_batch(rows=3,figsize = (5,5))


# In[ ]:


data.valid_ds.classes


# In[ ]:


kappa = KappaScore()
kappa.weights = "quadratic"
model = cnn_learner(data,models.resnet152, metrics = [accuracy,error_rate,kappa],callback_fns=ShowGraph)


# In[ ]:


model.summary()


# In[ ]:


model.lr_find()
model.recorder.plot(suggestion = True)


# In[ ]:


lr = 3e-3
model.fit_one_cycle(10,slice(lr))


# In[ ]:


model.unfreeze()
model.lr_find()
model.recorder.plot(suggestion = True)


# In[ ]:


# lr = 3e-3
model.fit_one_cycle(10,slice(1e-6,1e-8))


# In[ ]:


model.fit_one_cycle(10,slice(1e-6,1e-8))


# In[ ]:


model.save('stage-1-resnet152')


# In[ ]:


model.recorder.plot_losses()


# In[ ]:


interpreter = ClassificationInterpretation.from_learner(model)
interpreter.plot_confusion_matrix()


# In[ ]:


preds, _ = model.get_preds(ds_type=DatasetType.Test)
test_df.diagnosis = preds.argmax(1)
# test_df['id_code'] = test_df['id_code'].str.split(".", n = 1, expand = True) 
test_df.to_csv('submission.csv', index=False)
test_df.head()


# In[ ]:


# test_df['id_code'] = test_df['id_code'].str.split(".", n = 1, expand = True) 
# test_df.head()

