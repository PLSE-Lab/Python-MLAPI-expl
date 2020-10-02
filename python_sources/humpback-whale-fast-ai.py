#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np
import gc
gc.enable()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (8.0, 5.0)

import warnings
warnings.filterwarnings("ignore")

from fastai import *
from fastai.vision import *

from utils import *


# In[ ]:


path = Path('../input/humpback-whale-identification/')
path_test = Path('../input/humpback-whale-identification/test')
path_train = Path('../input/humpback-whale-identification/train')


# In[ ]:


train_df=pd.read_csv(path/'train.csv')
val_fns = {'69823499d.jpg'}


# In[ ]:


print("Train Shape : ",train_df.shape)


# In[ ]:


print("No of Whale Classes : ",len(train_df.Id.value_counts()))


# In[ ]:


train_df.Id.value_counts().head()


# In[ ]:


(train_df.Id == 'new_whale').mean()


# In[ ]:


(train_df.Id.value_counts() == 1).mean()


# 41% of all whales have only a single image associated with them.
# 
# 38% of all images contain a new whale - a whale that has not been identified as one of the known whales.

# In[ ]:


fn2label = {row[1].Image: row[1].Id for row in train_df.iterrows()}
path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)


# In[ ]:


gc.collect()


# In[ ]:


name = f'densenet169'

SZ = 224
BS = 64
NUM_WORKERS = 0
SEED=0


# In[ ]:


data = (
    ImageItemList
        .from_df(train_df[train_df.Id != 'new_whale'],path_train, cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder(path_test))
        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path=path)
).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3)


# In[ ]:


learn = create_cnn(data, models.densenet169, lin_ftrs=[2048], model_dir='../working/')
learn.clip_grad()


# In[ ]:


gc.collect()


# In[ ]:


SZ = 224 * 2
BS = 64 // 4
NUM_WORKERS = 0
SEED=0


# In[ ]:


df = pd.read_csv('../input/oversample-whale/oversampled_train_and_val.csv')


# In[ ]:


data = (
    ImageItemList
        .from_df(df, path_train, cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder(path_test))
        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path=path)
        .normalize(imagenet_stats)
)


# In[ ]:


learn = create_cnn(data, models.densenet169, lin_ftrs=[2048], model_dir='../working/')


# In[ ]:


learn.fit_one_cycle(1, slice(6.92E-06))


# In[ ]:


gc.collect()
learn.save('stage-1')


# In[ ]:


gc.collect()


# In[ ]:


preds, _ = learn.get_preds(DatasetType.Test)
preds = torch.cat((preds, torch.ones_like(preds[:, :1])), 1)
preds[:, 5004] = 0.06

classes = learn.data.classes + ['new_whale']


# In[ ]:


def top_5_preds(preds): return np.argsort(preds.numpy())[:, ::-1][:, :5]

def top_5_pred_labels(preds, classes):
    top_5 = top_5_preds(preds)
    labels = []
    for i in range(top_5.shape[0]):
        labels.append(' '.join([classes[idx] for idx in top_5[i]]))
    return labels

def create_submission(preds, data, name, classes=None):
    if not classes: classes = data.classes
    sub = pd.DataFrame({'Image': [path.name for path in data.test_ds.x.items]})
    sub['Id'] = top_5_pred_labels(preds, classes)
    sub.to_csv(f'{name}.csv', index=False)


# In[ ]:


create_submission(preds, learn.data, name, classes)

