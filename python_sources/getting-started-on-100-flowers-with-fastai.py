#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from fastai import *
from fastai.vision import *

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load data
# 
# This [dataset](https://www.kaggle.com/ianmoone0617/flower-goggle-tpu-classification) contains an already processed tfrecords data as csv and jpegs. Will be using it in this notebook.

# In[ ]:


df = pd.read_csv("../input/flower-goggle-tpu-classification/flowers_idx.csv")
label = pd.read_csv("../input/flower-goggle-tpu-classification/flowers_label.csv")


# In[ ]:


df.head()


# In[ ]:


label.head()


# In[ ]:


path = Path('/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/flowers_google/')
path_test = Path('/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/test/test/')


# In[ ]:


tfms = get_transforms(do_flip=True,max_rotate=0.1,max_lighting=0.15)
test = (ImageList.from_folder(path_test,extensions='.jpeg'))

data = (ImageList.from_df(df,path,folder='flowers_google',suffix='.jpeg',cols='id')
                .split_by_rand_pct(0.15)
                .label_from_df(cols='flower_cls')
                .transform(tfms)
                .add_test(test)
                .databunch(bs=32)
                .normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=4)


# # Train

# In[ ]:


arch = models.resnet50
learn = cnn_learner(data, arch, metrics=accuracy, model_dir='/kaggle/working')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.summary()


# In[ ]:


lr = 1e-2


# In[ ]:


learn.fit_one_cycle(6,lr,moms=(0.9,0.8))


# # Interpretation

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(12,figsize=(20,8))


# In[ ]:


interp.most_confused(min_val=3)


# # Prediction and submission

# In[ ]:


img = open_image('/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/test/test/d9cb87ad0.jpeg')
print(learn.predict(img)[0])
img


# In[ ]:


samp = pd.read_csv('/kaggle/input/flower-classification-with-tpus/sample_submission.csv')
n = samp.shape[0]
path_alltest = '/kaggle/input/flower-goggle-tpu-classification/flower_tpu/flower_tpu/test/test/'


# In[ ]:


for i in range(n):
    idc = samp.iloc[i][0]
    k = path_alltest + idc + '.jpeg'
    k = open_image(k)
    ans = learn.predict(k)[0]
    samp.loc[[i],1:] = str(ans)


# In[ ]:


samp.head(10)


# In[ ]:


lab = {}
for i in range(label.shape[0]):
  sha = label.iloc[i]
  lab[sha[1]]=int(sha[0])


# In[ ]:


samp.label.replace(lab,inplace=True)


# In[ ]:


samp.head()


# In[ ]:


samp.to_csv('submission.csv',index=False)


# In[ ]:




