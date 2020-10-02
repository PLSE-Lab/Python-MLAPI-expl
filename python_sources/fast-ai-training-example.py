#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Modules
import torch
import numpy as np
from fastai import *
from fastai.vision import *


# In[ ]:


## EfficientNet
# https://www.kaggle.com/chopinforest/efficientnetb4-fastai-blindness-detection/data
import sys
sys.path.append('../input/efficientnet-pytorch/efficientnet-pytorch/EfficientNet-PyTorch-master')


# In[ ]:


## Random State
SEED = 4
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# In[ ]:


## Prepare Data
import pandas as pd

train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

train.diagnosis.hist(); train.diagnosis.value_counts()


# Our models will have a tendency to overfit to classes 0 and 2. A stratified train/test split will provide a validation set with a similar distribution as the training data, but will improperly test the model's ability to classify the disease intensity across the board. We need a validation set that will represent the model's true discriminative power. 
# 
# Let's take 15% of each class stratified, but limit the over-represented classes (0 and 2) to 15% of the least-represented class (3). 

# In[ ]:


# Train/Test Split
limit = int(193 * 0.15)
zer = train[train.diagnosis == 0].sample(n=limit)
one = train[train.diagnosis == 1].sample(frac=0.15)
two = train[train.diagnosis == 2].sample(n=limit)
thr = train[train.diagnosis == 3].sample(frac=0.15)
fou = train[train.diagnosis == 4].sample(frac=0.15)
valid = pd.concat([zer, one, two, thr, fou])
train = train.drop(valid.index)

train['is_valid'] = False
valid['is_valid'] = True

train.diagnosis.hist(); valid.diagnosis.hist()

train = pd.concat([train, valid])


# In[ ]:


# Append File Extensions
append_ext = lambda fname: fname + '.jpeg' if '_' in fname else fname + '.png'
train.id_code = train.id_code.apply(append_ext)


# In[ ]:


# Extract
get_ipython().system(' tar -xzf ../input/aptos2019-224/train_images.tar.gz -C ./')


# In[ ]:


# Data Loader
# https://www.kaggle.com/drhabib/starter-kernel-for-0-79

bs = 128
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=180, max_zoom=1.1,
                      max_lighting=0.2, max_warp=0., p_affine=0.75, p_lighting=0.5)
data = (ImageList.from_df(train, path='./train_images', cols='id_code')
        .split_from_df()
        .label_from_df(label_cls=CategoryList) 
        .transform(tfms, padding_mode='zeros')
        .databunch(bs=bs)
        .normalize(imagenet_stats))


# In[ ]:


## Training
# QWK
import torch
from sklearn.metrics import cohen_kappa_score as cks

def qwk(y_pred, y_true):
    score = cks(torch.argmax(y_pred, dim=-1), y_true, weights='quadratic')
    return torch.tensor(score, device='cuda:0')


# In[ ]:


# Model
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_name('efficientnet-b0')
model.load_state_dict(torch.load('../input/efficientnet-pytorch/efficientnet-b0-08094119.pth'))
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, 5)  # Classification


# In[ ]:


# Model and Learner
from fastai.callbacks import *

sgd = partial(optim.SGD, momentum=0.9, weight_decay=1e-5, nesterov=True)
learn = Learner(
    data, model, opt_func=sgd, metrics=[qwk],
    path='./', callback_fns=[CSVLogger]).to_fp16()


# In[ ]:


# Train
epochs = 30
length = len(learn.data.train_dl) * epochs
anneal = [TrainingPhase(length).schedule_hp('lr', (1e-2, 1e-3), anneal=annealing_cos)]
callbacks = [GeneralScheduler(learn, anneal),
             SaveModelCallback(learn, monitor='qwk', name='best_qwk')]
learn.fit(epochs, callbacks=callbacks)


# In[ ]:


# Export Model
learn.load('best_qwk')
learn.export()


# In[ ]:


# Cleanup
import shutil
shutil.rmtree('./models')
shutil.rmtree('./train_images')

