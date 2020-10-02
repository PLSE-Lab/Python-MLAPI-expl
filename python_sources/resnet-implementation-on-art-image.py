#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input/dataset_updated/"))


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:



from fastai import *
from fastai.vision import *


# In[ ]:


PATH = "../input/dataset_updated/dataset_updated/"
PATH_OLD = "../input/dataset_updated/dataset_updated/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz=200


# In[ ]:


# GPU required
torch.cuda.is_available()


# In[ ]:


torch.backends.cudnn.enabled


# In[ ]:


os.listdir(PATH + 'training_set')


# In[ ]:


files = os.listdir(f'{PATH}training_set/engraving')[:5]
files


# In[ ]:


img = plt.imread(f'{PATH}training_set/engraving/{files[1]}')
plt.imshow(img);


# In[ ]:


img.shape


# In[ ]:


img[:4,:4]


# In[ ]:


# Fix to enable Resnet to live on Kaggle
cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# In[ ]:


#arch=resnet34
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(PATH, train='training_set', valid='validation_set', ds_tfms=tfms, size=sz, num_workers=0)


# In[ ]:


data.show_batch(rows=3, figsize=(6,6))


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir=MODEL_PATH)


# In[ ]:


learn.fit_one_cycle(6, slice(1e-5,3e-4), pct_start=0.05)
learn.recorder.plot()


# This is it!!

# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


preds,y,losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(data, preds, y, losses)
interp.plot_top_losses(9, figsize=(14,14))


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.most_confused(slice_size=10)

