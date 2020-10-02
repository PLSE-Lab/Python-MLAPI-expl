#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


path = Path('../input/aerial-cactus-identification/')
import pandas as pd


# In[ ]:


train = pd.read_csv('../input/aerial-cactus-identification/train.csv')
test = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')


# In[ ]:


np.random.seed(50)


# In[ ]:


tfms = get_transforms(do_flip = True,)


# In[ ]:


data = (ImageList.from_df(train , path = path/'train' , folder = 'train')
       .split_by_rand_pct(0.01)
        .label_from_df()
        .transform(tfms, size=128)
        .databunch()).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows = 3,figsize=(7,8))


# In[ ]:


learn = cnn_learner(data , models.resnet50 , metrics = error_rate)


# In[ ]:


learn.fit_one_cycle(4)
# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


test_data = ImageList.from_df(test, path=path/'test', folder='test')
data.add_test(test_data)


# In[ ]:


preds, _ = learn.get_preds(ds_type=DatasetType.Test)
test.has_cactus = preds.numpy()[:, 0]


# In[ ]:


test.to_csv("submit.csv", index=False)


# In[ ]:


preds


# In[ ]:




