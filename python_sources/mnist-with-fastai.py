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


# Import fastai libraries
from fastai.vision import *
from fastai.metrics import error_rate


# ## Getting the images.

# In[ ]:


get_ipython().system('cd /kaggle/input')
get_ipython().system('pwd')


# In[ ]:


# Uncompress the archives containing the images.
# The resulting datasets will be in:
#    /kaggle/input/trainingSet
#    /kaggle/input/testSet
# The trainingSet directory will contain subdirectories corresponding to the labels 0..9. 


get_ipython().system('tar -zxf ../input/mnistasjpg/trainingSet.tar.gz')
get_ipython().system('tar -xxf ../input/mnistasjpg/testSet.tar.gz ')
get_ipython().system('mv trainingSet ../input')
get_ipython().system('mv testSet ../input')
get_ipython().system('ls -li /kaggle/input')

# Note: this could have been done as well using fastai utility method untar_data().


# ## Constants

# In[ ]:


batchsize = 64
path_training = Path('/kaggle/input/trainingSet')
path_test = Path('/kaggle/input/testSet')


# ## Read training data.

# In[ ]:


data = ImageDataBunch.from_folder(path_training, valid_pct=0.15, seed=123, 
                                  test=path_test, size=28,
                                  bs=batchsize, ds_tfms=get_transforms(do_flip=False))
data.normalize(mnist_stats)
data.show_batch(rows=3, figsize=(7, 6))
print('Available labels: {}'.format(data.classes))


# ## Training.

# In[ ]:


model = models.resnet34 # Results in error_rate 0.0046 with 4 epochs plus full retraining for 12 epochs with max_lr=slice(1e-4, 1e-3)
# model = models.resnet50 # 0.052 with 4 epochs plus full retraining for 12 epochs with max_lr=slice(1e-4, 1e-3)


# In[ ]:


learn = cnn_learner(data, model, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1-resnet34-4-epochs')


# ## Finetuning.

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.load('stage-1-resnet34-4-epochs')
learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(12, max_lr=slice(1e-4, 1e-3))


# In[ ]:


learn.save('resnet34-4-12-1e-4-1e-3')


# ## Results.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(36, figsize=(12,12))


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.most_confused(min_val=2)


# ## Evaluate on testset.

# In[ ]:


test_results = pd.DataFrame(columns=['ImageId','Label'])
for i in range(1, 28000 + 1):
    img = open_image(path_test/'img_{}.jpg'.format(i))
    prediction = learn.predict(img)[1].item()
    test_results.loc[i] = [i, prediction]


# ## Results sanity check.

# In[ ]:


test_results.head()


# ## Write testset results to file.

# In[ ]:


test_results.to_csv('submission.csv', index=False)

