#!/usr/bin/env python
# coding: utf-8

# **PLEASE UPVOTE IF FOUND INTERESTING**

# Kernal for Preprocessing INKML to PNG Image to available [here](https://www.kaggle.com/kalikichandu/preprossing-inkml-to-png-files)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


print(os.listdir("../input/handwritten-math-symbols-dataset/handwrittenmathsymbols/data/extracted_images"))


# Note - Here I have uploded the Zip Version of DataSet as private Dataset

# In[ ]:


get_ipython().system('pip install split_folders    # Library to split Train and valid Image sets in ImageNet style')


# In[ ]:


import os
import numpy as np
from tqdm import tqdm
from fastai import *
import torch
from fastai.vision import *
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.callbacks import *
import cv2
import pandas as pd
import split_folders


# In[ ]:


image_load_size = 64
bs = 24


# In[ ]:


def seed_everything(seed):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.backends.cudnn.deterministic = True    

SEED = 999
seed_everything(SEED)


# In[ ]:


split_folders.ratio('../input/handwritten-math-symbols-dataset/handwrittenmathsymbols/data/extracted_images', output="../final_output_images", seed=SEED, ratio=(.8, .2)) # default va


# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=False, max_lighting=0.1, max_zoom=1.05,
                      max_warp=0.,
                      xtra_tfms=[rand_crop(), rand_zoom(1, 1.5),
                                 symmetric_warp(magnitude=(-0.2, 0.2))])


# In[ ]:


data = (ImageList.from_folder(path='../final_output_images')
        .split_by_folder(train='train',valid='val')
        .label_from_folder()
        .transform(tfms,size = image_load_size,resize_method=ResizeMethod.SQUISH)
        .databunch(path='.',bs=bs)    
        .normalize(imagenet_stats)
)


# In[ ]:


data.show_batch(3, figsize=(6,6), hide_axis=False)


# In[ ]:


len(data.classes)


# In[ ]:


data.classes


# In[ ]:


class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
        return F_loss.mean()


# In[ ]:


model = cnn_learner(data,models.densenet161, metrics = [accuracy,error_rate],callback_fns=[partial(SaveModelCallback, monitor='accuracy', name='best_model')])
model.loss_func = FocalLoss()
model.summary()


# In[ ]:


model.lr_find()
model.recorder.plot(suggestion = True)


# In[ ]:


lr = 2e-3
model.fit_one_cycle(5,slice(lr))


# In[ ]:


model.unfreeze()
model.lr_find()
model.recorder.plot(suggestion = True)


# In[ ]:


model.fit_one_cycle(5,slice(1e-6,lr/10))


# In[ ]:


model.recorder.plot_metrics()


# In[ ]:


model.load('best_model')


# In[ ]:


valid_loss_save_model,accuracy_save_model, error_rate_save_model = model.validate(model.data.valid_dl)
print('valid_loss:', valid_loss_save_model, 'accuracy:', float(accuracy_save_model),'error_rate:',float(error_rate_save_model))


# In[ ]:


interpreter = ClassificationInterpretation.from_learner(model)
interpreter.plot_confusion_matrix(figsize = (20,20))


# In[ ]:


interpreter.most_confused(min_val=50)


# In[ ]:


print(os.listdir('../final_output_images/val/A'))


# In[ ]:


open_image('../final_output_images/val/A/exp3159.jpg')


# In[ ]:


pred = model.predict(open_image('../final_output_images/val/A/exp3159.jpg'))
print(pred[0])


# Comment below incase if any clarification is needed.
