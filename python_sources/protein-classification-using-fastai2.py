#!/usr/bin/env python
# coding: utf-8

# # Import modules

# fastai is a deep learning library which is easier to use and for people from different backgrounds. It consists of high level API components that can be used to achieve state of the art results quiculy and easily. In this competetion, we experiment with fastai library to see if we can achieve high performance model. 
# 
# https://arxiv.org/abs/2002.04688

# In[ ]:


get_ipython().system('pip install fastai2 -q')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import pandas as pd
import numpy as np

from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *
from fastai2.vision.widgets import *
from fastai2.vision import *
from fastai2.callback.cutmix import *


# # Data

# In[ ]:


DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'

TRAIN_DIR = DATA_DIR + '/train'                           
TEST_DIR = DATA_DIR + '/test'                             

TRAIN_CSV = DATA_DIR + '/train.csv'                       
TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv' 


# #### Create a new column with image paths

# In[ ]:


train_df = pd.read_csv(TRAIN_CSV)
train_df['imgPath'] = train_df.apply(lambda x : os.path.join(TRAIN_DIR,str(x['Image'])+'.png'),axis=1)
train_df.head()


# # Stratification

# Here we implement stratifed split of the training data as compared to random split. This implementation was taken from https://www.kaggle.com/ronaldokun/multilabel-stratification-cv-and-ensemble

# In[ ]:


split_df = pd.get_dummies(train_df.Label.str.split(" ").explode())
split_df = split_df.groupby(split_df.index).sum()
split_df.head()


# In[ ]:


X, y = split_df.index.values, split_df.values


# In[ ]:


from skmultilearn.model_selection import IterativeStratification

nfolds = 5

k_fold = IterativeStratification(n_splits=nfolds, order=1)

splits = list(k_fold.split(X, y))

fold_splits = np.zeros(train_df.shape[0]).astype(np.int)

for i in range(nfolds):
    fold_splits[splits[i][1]] = i


# In[ ]:


train_df['Split'] = fold_splits


# 'is_valid' column will be used to the train and validation split when creating the datablocks. 

# In[ ]:


def get_fold(fold):
    train_df['is_valid'] = False
    train_df.loc[train_df.Split == fold, 'is_valid'] = True
    return train_df


# In[ ]:


train_df.head()


# In[ ]:


train_df = get_fold(0)
train_df = train_df.drop(['Split'],axis=1)
train_df.head()


# In[ ]:


test_df = pd.read_csv(TEST_CSV)
test_df['imgPath'] = test_df.apply(lambda x : os.path.join(TEST_DIR,str(x['Image'])+'.png'),axis=1)
test_df.head()


# In[ ]:


labels = {
    0: 'Mitochondria',
    1: 'Nuclear bodies',
    2: 'Nucleoli',
    3: 'Golgi apparatus',
    4: 'Nucleoplasm',
    5: 'Nucleoli fibrillar center',
    6: 'Cytosol',
    7: 'Plasma membrane',
    8: 'Centrosome',
    9: 'Nuclear speckles'
}


# # Helper functions

# In[ ]:


def encode_label(label):
    target = torch.zeros(10)
    for l in str(label).split(' '):
        target[int(l)] = 1.
    return target

def decode_target(target, text_labels=False, threshold=0.5):
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return ' '.join(result)


# # Dataloaders

# ### Data augmentation

# In[ ]:


aug_tfms = aug_transforms(mult=1.0, 
               do_flip=True, 
               flip_vert=False, 
               max_rotate=10.0, 
               max_zoom=1.1, 
               max_lighting=0.5, 
               max_warp=0.2, 
               p_affine=0.75, 
               p_lighting=0.75, 
               xtra_tfms=RandomErasing(p=1., max_count=6), 
               size=224, 
               mode='bilinear', 
               pad_mode='reflection', 
               align_corners=True, 
               batch=False, 
               min_scale=0.75)


# #### functions for getting image path and labels in dataloaders

# In[ ]:


def get_x(r): return r['imgPath']
def get_y(r): return r['Label'].split(' ')
def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train,valid


# In[ ]:


dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter, 
                   get_x = get_x, get_y = get_y,
                   item_tfms=Resize(460),
                   batch_tfms=aug_tfms)


# In[ ]:


dls = dblock.dataloaders(train_df)


# ### Show a batch of train data

# #### Random erasing augmentation have erased regions of the image and replaced them with gaussian noise

# In[ ]:


dls.train.show_batch(max_n=9)


# # Multi-label metrics

# As this is a multi-label classification problem we cannot use softwax but use sigmoid activiation with threshold. The threshold is a hyperparameter which determines the prediction of the labels. We have to experiment with different threshold value to find the optimal one. Here we are using two metric accuracy_multi and F1_multi and both takes threshold as input parameter. 

# In[ ]:


def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()


# In[ ]:


def F_score(output, label, threshold=0.2, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)


# # CNN learner

# ### 1. Using cutmix data augmentation

# Cutmix is a data augmentation technique in which parts of images are cut and mixed with another image. It comes from Cutout and Mixup. 
# https://arxiv.org/pdf/1905.04899.pdf

# In[ ]:


cutmix = CutMix(1.0)
learn = cnn_learner(dls, resnet18, 
                    metrics=[partial(accuracy_multi, thresh=0.2),partial(F_score, threshold=0.2)],
                    cbs=cutmix,
                   callback_fns=[partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=3)])


# In[ ]:


learn._do_begin_fit(1)
learn.epoch,learn.training = 0,True
learn.dl = dls.train
b = dls.one_batch()
learn._split(b)
learn('begin_batch')
_,axs = plt.subplots(3,3, figsize=(9,9))
dls.show_batch(b=(cutmix.x,cutmix.y), ctxs=axs.flatten())


# In[ ]:


learn.fine_tune(3, base_lr=3e-3, freeze_epochs=2)


# ### 2. Using Mixup data augmentation

# In[ ]:


mixup = MixUp(0.4) 
learn = cnn_learner(dls, resnet18, 
                    metrics=[partial(accuracy_multi, thresh=0.2),partial(F_score, threshold=0.2)],
                    cbs=mixup,
                   callback_fns=[partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=3)])


# In[ ]:


learn._do_begin_fit(1)
learn.epoch,learn.training = 0,True
learn.dl = dls.train
b = dls.one_batch()
learn._split(b)
learn('begin_batch')
_,axs = plt.subplots(3,3, figsize=(9,9))
dls.show_batch(b=(mixup.x,mixup.y), ctxs=axs.flatten())


# In[ ]:


learn.fine_tune(3, base_lr=3e-3, freeze_epochs=2)


# ### 3.Without cutmix or mixup

# In[ ]:


learn = cnn_learner(dls, resnet18, 
                    metrics=[partial(accuracy_multi, thresh=0.2),partial(F_score, threshold=0.2)],
                   callback_fns=[partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=3)])


# In[ ]:


learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)


# # Selecting the suitable threshold for multi-metrics

# In[ ]:


preds,targs = learn.get_preds()


# In[ ]:


xs = torch.linspace(0.05,0.95,29)
accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False) for i in xs]
fscores = [F_score(preds, targs, threshold=i, beta=1) for i in xs]


# In[ ]:


plt.plot(xs,accs);


# In[ ]:


plt.plot(xs,fscores);


# Looking at the graph above 0.2 seems to be thr right threshold

# # Training

# In[ ]:


learn = cnn_learner(dls, resnet18, 
                    metrics=[partial(accuracy_multi, thresh=0.2),partial(F_score, threshold=0.2)],
                   callback_fns=[partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=3)])


# In[ ]:


learn.fine_tune(20, base_lr=3e-3, freeze_epochs=4)


# # Model performance

# In[ ]:


learn.show_results()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(5, nrows=1)


# In[ ]:


learn.export('export.pkl')


# # Load the test data and do predictions

# In[ ]:


dl = learn.dls.test_dl(test_df)


# ### Test time augmentation during prediction of test data

# In[ ]:


preds,targs = learn.tta(dl=dl)


# # Prediction and submission

# using threshold = 0.5

# In[ ]:


predictions = [decode_target(x, threshold=0.5) for x in preds]


# In[ ]:


submission_df = pd.read_csv(TEST_CSV)
submission_df.Label = predictions
submission_df.head()


# In[ ]:


sub_fname = 'submission_fastai_v6_1.csv'


# In[ ]:


submission_df.to_csv(sub_fname, index=False)


# Using threshold = 0.2

# In[ ]:


predictions = [decode_target(x, threshold=0.2) for x in preds]


# In[ ]:


submission_df = pd.read_csv(TEST_CSV)
submission_df.Label = predictions
submission_df.head()


# In[ ]:


sub_fname = 'submission_fastai_v6_2.csv'


# In[ ]:


submission_df.to_csv(sub_fname, index=False)


# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[ ]:


import jovian


# In[ ]:


project_name='protein-advanced'


# In[ ]:


jovian.commit(project=project_name, environment=None)


# In[ ]:




