#!/usr/bin/env python
# coding: utf-8

# First let's import the necessary libraries

# In[ ]:


import os
import numpy
from fastai.vision import *
import pandas as pd


# Let's check the data

# In[ ]:


path = Path('../input')
path.ls()


# In[ ]:


train_dir = path/'train'/'train'
test_dir = path/'test'/'test'
train_labels = path/'train.csv'
output = path/'sample_submission.csv'


# In[ ]:


np.random.seed(26)


# In[ ]:


data = (ImageList.from_csv(path=path, csv_name='train.csv', folder='train/train')
        .split_by_rand_pct()
        .label_from_df(cols='has_cactus')
        .add_test(ImageList.from_csv(path=path, csv_name='sample_submission.csv', 
                                            folder='test/test'))
        .transform(get_transforms(), size=224)
        .databunch(bs=32, num_workers=0)
        .normalize(imagenet_stats))


# In[ ]:


# data.show_batch(rows=3, figsize=(10,10))


# In[ ]:


data.classes, data.classes, data.train_ds, data.test_ds


# In[ ]:


from fastai.vision.learner import create_cnn, models


# In[ ]:


arch = cnn_learner(data, base_arch=models.resnet34, model_dir='/tmp/models',
                   metrics=accuracy)


# In[ ]:


arch.lr_find()


# In[ ]:


arch.recorder.plot()


# In[ ]:


epoch = 10
lr = 1e-3

arch.fit_one_cycle(epoch, slice(lr))


# In[ ]:


arch.save('model-v1')


# In[ ]:


arch.load('model-v1')


# In[ ]:


arch.unfreeze()
arch.lr_find()
arch.recorder.plot()


# In[ ]:


arch.fit_one_cycle(3, max_lr=slice(lr/1000,lr/100))


# In[ ]:


prediction, y = arch.get_preds(ds_type=DatasetType.Test)


# In[ ]:


test_data = pd.read_csv(output)
test_data.has_cactus = prediction.numpy()[:, 0]
test_data.head(10)


# In[ ]:


test_data.to_csv('submission.csv', index=False)

