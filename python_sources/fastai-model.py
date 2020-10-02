#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


from fastai.vision import *


# In[ ]:


get_ipython().system('ls ../input/plant-seedlings-classification/train')


# In[ ]:


path = Path('../input/plant-seedlings-classification/train')


# In[ ]:


path.ls()


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".",test='../test', valid_pct=0.2,ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3,fig_size=(7,8))


# In[ ]:


data.classes, data.c , len(data.train_ds) , len(data.valid_ds) , len(data.test_ds)


# **train the model**

# In[ ]:


learn= cnn_learner(data,models.resnet50,metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.model_dir='/kaggle/working/'


# In[ ]:


learn.save('stage_1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5,slice(1e-3))


# In[ ]:


learn.save('stage_2')


# ***interpretation***

# In[ ]:


learn.load('stage_2');


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:



interp.plot_confusion_matrix(figsize=(10,10))


# In[ ]:


interp.plot_top_losses(4,figsize=(10,10))


# In[ ]:


preds,y=learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


preds = np.argmax(preds, axis = 1)
preds_classes = [data.classes[i] for i in preds]


# In[ ]:


submission = pd.DataFrame({ 'file': os.listdir('../input/plant-seedlings-classification/test'), 'species': preds_classes })
submission.to_csv('test_classification_results.csv', index=False)


# In[ ]:


submission


# <a href="test_classification_results.csv"> Download File </a>

# In[ ]:




