#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from pathlib import Path

from fastai import *
from fastai.vision import *
from fastai.callbacks import ReduceLROnPlateauCallback, EarlyStoppingCallback, SaveModelCallback

import os


# In[2]:


os.listdir('../input/dermmel/DermMel/')


# In[3]:


data_path = Path('../input/dermmel/DermMel/')


# In[4]:


transforms = get_transforms(do_flip = True, 
                            flip_vert = True, 
                            max_rotate = 355.0, 
                            max_zoom = 1.5, 
                            max_lighting = 0.3, 
                            max_warp = 0.2, 
                            p_affine = 0.75, 
                            p_lighting = 0.75)


# In[5]:


data = ImageDataBunch.from_folder(data_path,
                                  valid_pct = 0.15,
                                  size = 200,
                                  bs = 64,
                                  ds_tfms = transforms
                                 )

data.normalize(imagenet_stats)


# In[6]:


data.classes


# In[7]:


data.show_batch(rows = 5, figsize = (12, 12))


# In[8]:


learn = cnn_learner(data, models.resnet152 , metrics = [accuracy], model_dir = '/tmp/model/')


# In[9]:


reduce_lr_pateau = ReduceLROnPlateauCallback(learn, patience = 10, factor = 0.2, monitor = 'accuracy')

#early_stopping = EarlyStoppingCallback(learn, monitor = 'accuracy', patience = 6)

save_model = SaveModelCallback(learn, monitor = 'accuracy', every = 'improvement')

callbacks = [reduce_lr_pateau, save_model]


# In[10]:


learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion = True)


# In[ ]:


min_grad_lr = learn.recorder.min_grad_lr

learn.fit_one_cycle(100, min_grad_lr, callbacks = callbacks, wd = 1e-3)


# In[ ]:


learn.save('model')


# In[ ]:


learn.load('model')


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(9, figsize = (12, 12), heatmap = False)


# In[ ]:


interp.most_confused()


# In[ ]:


predictions, y, loss = learn.get_preds(with_loss = True)

acc = accuracy(predictions, y)


# In[ ]:


print('Accuracy: {0}'.format(acc))

