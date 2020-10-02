#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fastai.vision import *
from fastai.metrics import error_rate, accuracy

print(os.listdir("../input"))


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


PATH = "../input/"
MODEL_PATH = "/tmp/model/"


# In[ ]:


df = pd.read_csv('../input/labels.csv')
df.head(10)


# In[ ]:


print(df['breed'].value_counts().sort_values(ascending=False))


# In[ ]:


plt.figure(figsize=(12,8))
plt.hist(df['breed'].value_counts().sort_values(ascending=False))
plt.show()


# In[ ]:


tfms = get_transforms(max_rotate=25); len(tfms)


# In[ ]:


data = ImageDataBunch.from_csv(PATH, folder='train', test='test', suffix='.jpg', ds_tfms=tfms,
                               csv_labels='labels.csv', fn_col=0, label_col=1, 
                               size=128, bs=64).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=4, figsize=(12,8))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=[accuracy], model_dir=MODEL_PATH)


# In[ ]:


learn.model


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(3e-2,1e-2))


# In[ ]:


learn.save('/tmp/model/stage-1-34')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


#learn.load('/tmp/model/stage-1')
learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.save('/tmp/model/stage-2-34')


# In[ ]:


data = ImageDataBunch.from_csv(PATH, folder='train', test='test', suffix='.jpg', ds_tfms=tfms,
                               csv_labels='labels.csv', fn_col=0, label_col=1, 
                               size=224, bs=16).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=4, figsize=(12,8))


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=[accuracy], model_dir=MODEL_PATH)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, max_lr=slice(1e-3,1e-2))


# In[ ]:


learn.save('/tmp/model/stage-1-50')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, max_lr=slice(1e-5,1e-4))


# In[ ]:


learn.save('/tmp/model/stage-2-50')


# In[ ]:


predictions = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


predictions[0][0]


# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()


# In[ ]:


submission = sample_submission.copy()
for i in range(len(submission)):
    submission.iloc[i, 1:] = predictions[0][i].tolist()
submission.head()


# In[ ]:




