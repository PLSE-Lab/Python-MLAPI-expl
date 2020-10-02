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


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *


# In[ ]:


path = Path('/kaggle/input/blood-cells/dataset2-master/dataset2-master/images')
path.ls()


# Notice how the 'label' of the Y (the answer) is named after the folder. So all that belongs to 'MONOCYTE' is placed on that particular folder. In this case we want to use ImageDataBunch.from_folder to create the data object

# In[ ]:


(path/'TRAIN').ls()


# In[ ]:


data = ImageDataBunch.from_folder(path, train='TRAIN', test='TEST', valid_pct=0.20,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# let's see what the structure of the data look like:

# In[ ]:


data


# In[ ]:


data.show_batch(rows=3, figsize=(10,10))


# In[ ]:


data.c, len(data.train_ds), len(data.valid_ds), len(data.classes), len(data.test_ds)


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.model_dir=Path('/kaggle/working')
learn.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9,figsize=(20,11))


# In[ ]:


doc(interp.plot_top_losses)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=100)


# In[ ]:


interp.most_confused(min_val = 2)


# In[ ]:


help(interp.confusion_matrix)


# Unfreezeing, fine tuning and learning rates

# In[ ]:


learn.unfreeze()


# why is the result is better than previous fit cycle, this should not be the case. Or probably Fast.ai course by Jeremy has some concept problem. Probably it turns out the certain images has a different effect on this whole learning cycle ???

# In[ ]:


learn.fit_one_cycle(1)


# Loading the previous trained model again, we want to find the optimum learning rate.

# In[ ]:


learn.load('stage-1')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# Looking at the error rate above around 2% it seems the model is getting better than the previous fir cycle.

# In[ ]:


learn.save('model_resnet34')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=100)


# In[ ]:


interp.most_confused()


# the above result looks very good? it seems to cut the error by half of what we get from the previous learning cycle without any learning rate optimization ... or is this always true. There are scenarios that i haven't even tried, for example what if I just kept on iterating or increase the epoch cycle from 4 to say 10, would the error get decreased? Until the next kernel ...

# In[ ]:


learn


# In[ ]:


doc(learn.predict)


# In[ ]:


doc(ImageDataBunch.from_folder)


# In[ ]:




