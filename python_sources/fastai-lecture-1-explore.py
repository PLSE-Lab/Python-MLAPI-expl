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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


bs = 64


# In[ ]:


path = Path("../input/images/")


# In[ ]:


path_img = path/'images'


# In[ ]:


fnames = get_image_files(path_img)
fnames[:5]


# In[ ]:


np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'


# In[ ]:


data = ImageDataBunch.from_name_re('.', fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs,num_workers=0
                                  ).normalize(imagenet_stats)


# In[ ]:


data.device


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


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


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.load('stage-1');


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# ### Min lr value is around 3.6 e-05, choose a max lr of 1e-04 and min around 

# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:




