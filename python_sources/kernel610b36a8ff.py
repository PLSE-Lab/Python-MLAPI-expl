#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pathlib
from fastai.vision import *
from fastai.metrics import error_rate
print(os.listdir("../input/fruits"))

get_ipython().system('cp -r ../input/fruits/fruits-360_dataset/fruits-360 ../working/data')
# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('mv ../working/data/Training ../working/data/train')
get_ipython().system('mv ../working/data/Test ../working/data/valid')


# In[ ]:


print(os.listdir("../working/data"))


# In[ ]:


from pathlib import Path
path = Path("../working/data")


# In[ ]:


tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path,ds_tfms=tfms, size=126).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(3)
learn.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


doc(interp.plot_top_losses)


# In[ ]:


interp.plot_confusion_matrix(figsize=(30,30), dpi=120)


# In[24]:


learn.unfreeze()


# In[25]:


learn.fit_one_cycle(1)


# In[32]:


learn.lr_find()


# In[33]:


learn.recorder.plot()


# In[34]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-4,1e-3))


# In[35]:


learn.save('satage-a')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(60,60), dpi=120)


# In[36]:


learn.export()


# In[38]:


get_ipython().system('cp -r ../input/xxapple ../working/apple')


# In[39]:


print(os.listdir('../working/apple'))


# In[40]:


img = open_image('../working/apple/fresh-apple-500x500.jpg')
img


# In[41]:


learn = load_learner(path)


# In[43]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class

