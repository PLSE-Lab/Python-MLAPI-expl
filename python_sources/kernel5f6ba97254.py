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


import pandas as pd


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


path = Path('../input/')
train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')


# In[ ]:


train, test = [ImageList.from_df(df, path=path, cols='img_file', folder=folder) 
               for df, folder in zip([train_df, test_df], ['train', 'test'])]

tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=0.20, max_zoom=2, max_lighting=0.1)

data = (train.split_by_rand_pct(0.1, seed=42)
        .label_from_df(cols='class')
        .add_test(test)
        .transform(tfms, size=224)
        .databunch(path=Path('.'), bs=64).normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=4)


# In[ ]:


print(data.classes)
len(data.classes), data.c


# In[ ]:


learn = cnn_learner(data, models.densenet201, metrics=[accuracy, error_rate])


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-06, 1e-02))


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.load('stage-1')
learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10, max_lr=slice(1e-07, 1e-04))


# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.unfreeze()
learn.lr_find(start_lr=1e-10)


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit(epochs=10, lr=1e-4)


# In[ ]:


learn.save('stage-3')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit(epochs=5, lr=1e-5)


# In[ ]:


learn.save('stage-4')


# In[ ]:


test_preds = learn.TTA(ds_type=DatasetType.Test)
test_df['class'] = np.argmax(test_preds[0], axis=1) + 1
test_df.head()


# In[ ]:


test_df.to_csv('submission.csv', index=False) 


# In[ ]:


learn.load('stage-4')
learn.fit(epochs=10, lr=1e-7)


# In[ ]:




