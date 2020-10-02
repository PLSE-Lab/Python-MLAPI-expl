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
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


os.listdir('../input/pressure-ulcers/pressure_ulcers')
# !rm ../input/pressure_ulcers/*.jpg


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from fastai import *
from fastai.vision import *
from fastai.vision.data import ImageDataBunch, ImageList
from fastai.vision.transform import *
from fastai.vision.learner import cnn_learner
from fastai.vision.models import resnet50
from fastai.metrics import accuracy
from fastai.callbacks import MixUpCallback, SaveModelCallback
from fastai.train import ShowGraph


# In[ ]:


tfms = get_transforms(flip_vert=True, max_zoom=1.2, max_rotate=90)
tfms[0].pop(0)
tfms[1].pop(0)
tfms


# In[ ]:


path = '../input/pressure-ulcers/pressure_ulcers'

data = (ImageList.from_folder(path+'/train/')
                .split_by_rand_pct(0.2)
                .label_from_folder()
                .transform(tfms, size=256)
                .databunch(bs=32)
                .normalize())

test_data = (ImageList.from_folder(path+'/test/'))
data.add_test(test_data)


# In[ ]:


data


# In[ ]:


data.show_batch()


# In[ ]:


learner = cnn_learner(data, resnet50, metrics=accuracy,
                      wd=0.0009,
                      model_dir=os.getcwd()+'models',
                     callback_fns=[ShowGraph, SaveModelCallback])


# In[ ]:


# learner.lr_find()


# In[ ]:


# learner.recorder.plot(suggestion=True)


# In[ ]:


learner.fit_one_cycle(cyc_len=20, max_lr=1e-3)


# # valid without tta

# In[ ]:


# learner.show_results(rows=4)


# In[ ]:


# preds,y,losses = learner.get_preds(with_loss=True)
# interp = ClassificationInterpretation(learner, preds, y, losses)


# In[ ]:


# interp.plot_top_losses(9, figsize=(7,7))


# In[ ]:


# interp.plot_confusion_matrix()


# # result of train_dataset

# In[ ]:


# learner.show_results(ds_type=DatasetType.Train)


# In[ ]:


# preds,y,losses = learner.get_preds(ds_type=DatasetType.Train, with_loss=True)
# interp = ClassificationInterpretation(learner, preds, y, losses,ds_type=DatasetType.Train)
# interp.plot_top_losses(16)


# In[ ]:


# interp.plot_confusion_matrix()


# # add tta on valid

# In[ ]:


# preds,y,losses = learner.TTA(with_loss=True)
# interp = ClassificationInterpretation(learner, preds, y, losses)
# interp.plot_top_losses(9)


# In[ ]:


# interp.plot_confusion_matrix()


# # with mix up

# In[ ]:


# learner = cnn_learner(data, resnet50, metrics=accuracy,
#                       wd=0.0009,
#                       model_dir=os.getcwd()+'models',
#                      callback_fns=[ShowGraph, SaveModelCallback]).mixup()


# In[ ]:


# learner.fit_one_cycle(cyc_len=20, max_lr=1e-3)


# In[ ]:


# preds,y,losses = learner.get_preds(with_loss=True)
# interp = ClassificationInterpretation(learner, preds, y, losses)
# interp.plot_top_losses(9)


# In[ ]:


# interp.plot_confusion_matrix()


# # test

# In[ ]:


preds, y, losses = learner.get_preds(DatasetType.Test, with_loss=True)
interp = learner.interpret(DatasetType.Test, tta=True)
interp


# In[ ]:


get_ipython().run_line_magic('pinfo', 'interp.most_confused')


# # image cleaner

# In[ ]:


# from fastai.widgets import *


# In[ ]:


# ds, idxs = DatasetFormatter().from_similars(learner)


# In[ ]:


# ImageCleaner(ds, idxs, '/kaggle/working/')


# # downloading image

# In[ ]:


# os.makedirs('kaggle/working/0', exist_ok=True)


# In[ ]:


# os.makedirs('kaggle/working/1', exist_ok=True)
# files = download_google_images('kaggle/working/4', 'pressure ulcers Deep Tissue Injury', size='>400*300', n_images=100)


# In[ ]:


get_ipython().system('pip install selenium')


# In[ ]:




