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


from fastai import *
from fastai.vision import *
from fastai.widgets import *
import numpy as np


# In[ ]:


path = '../input/complete-pokemon-image-dataset/pokemon'


# In[ ]:


np.random.seed(42)
data = (ImageList.from_folder(path).split_by_rand_pct()
        .label_from_folder().transform(get_transforms(), size=128).databunch().normalize(imagenet_stats))


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[ ]:


learn = cnn_learner(data, models.resnet101, metrics=accuracy, model_dir="/tmp/model/")


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('Pokemons_Resnet101_stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-5,1e-4))


# In[ ]:


learn.save('Pokemons_Resnet101_stage-2')


# In[ ]:


data = ImageDataBunch.from_folder(path, train=".", 
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(flip_vert=False),
                                  size=128,bs=64, 
                                  num_workers=0).normalize(imagenet_stats)


# In[ ]:


learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-6))


# In[ ]:


learn.save('Pokemon_Resnet101_stage-3')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()

learn.recorder.plot_losses()
# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.most_confused()


# In[ ]:


learn.load('Pokemon_Resnet101_stage-3')


# In[ ]:


learn.export('/tmp/model/pokemon.pkl')


# In[ ]:


defaults.device = torch.device('cpu')


# In[ ]:


img1 = open_image('../input/complete-pokemon-image-dataset/pokemon/Alakazam(Mega)/Alakazam(Mega)_20.jpg')
img1


# In[ ]:


img2 = open_image('../input/complete-pokemon-image-dataset/pokemon/Hitmonchan/Hitmonchan_27.jpg')
img2


# In[ ]:


img3 = open_image('../input/complete-pokemon-image-dataset/pokemon/Cacnea/Cacnea_8.jpg')
img3


# In[ ]:


img4 = open_image('../input/complete-pokemon-image-dataset/pokemon/Pikachu/Pikachu_5.jpg')
img4


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img1)
pred_class


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img2)
pred_class


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img3)
pred_class


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img4)
pred_class

