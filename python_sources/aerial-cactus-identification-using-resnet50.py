#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai import *
from fastai.metrics import error_rate
import pandas as pd
import torch


# In[ ]:


path ="../input/"
train_df=pd.read_csv(path+"train.csv")
test_df=pd.read_csv(path+"sample_submission.csv")


# In[ ]:


bs = 128
data = ImageDataBunch.from_csv(path=path, folder='train/train', csv_labels='train.csv', ds_tfms=get_transforms(), size=32, bs=bs).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")


# In[ ]:


learn.fit_one_cycle(6)


# In[ ]:


learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(3e-5,3e-4))


# In[ ]:





# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp.plot_top_losses(4, figsize=(6,6))


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


a,b,c=learn.predict(open_image("../input/test/test/000940378805c44108d287872b2f04ce.jpg"))
print(c)
print(c[1].numpy())


# In[ ]:


test_df.head()


# In[ ]:


def pred(name):
    a,b,c=learn.predict(open_image("../input/test/test/"+name))
    return c[1].numpy()


# In[ ]:


test_df["has_cactus"]=test_df["id"].apply(lambda x:pred(x))


# In[ ]:


test_df.to_csv('submission.csv',index=False)


# In[ ]:




