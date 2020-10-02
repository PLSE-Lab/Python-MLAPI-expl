#!/usr/bin/env python
# coding: utf-8

# ## Sports Ball Image Recognition

# With deep learning, this model observes images and identifies which sport a ball is from. Transfer learning was used on a convoluted neural network trained on ResNet-34. Code was referenced from a tutorial by Francisco Ingham and Jeremy Howard: https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastai.vision import *

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


proj_path = '/kaggle/input/sports-ball-images/'

folders = ['airhockeypucks', 'baseballs', 'basketballs', 'bowlingballs', 'cricketballs', 'footballs', 'golfballs', 'hockeypucks', 'lacrosseballs', 'poolballs', 'rugbyballs', 'soccerballs', 'softballs', 'tennisballs', 'volleyballs']
for i in folders:
    path = Path(proj_path)
    (path/i).mkdir(parents=True, exist_ok=True)
    
p_path = Path(proj_path)


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train='.', valid_pct=.2, ds_tfms=get_transforms(), 
                                  size=224, num_workers=4).normalize(imagenet_stats)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(7,8))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# I will be using a convolutional neural net learner and transfer learning on the ResNet34 pre-trained model. Transfer learning allows me to fix the end of the pre-trained model for my own use. The metric used is error rate, which is 1 - accuracy.

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


# 8 epochs
learn.fit_one_cycle(8)


# Saving the newly built model.

# In[ ]:


learn.model_dir = "/kaggle/working"
learn.save('model1_34', return_path=True)


# Unfreezing to retrain the model.

# In[ ]:


learn.unfreeze()


# To find an optimal learning rate.

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# Testing to see if changing the learning rate gets better accuracies. We want a steep learning rate: about 1e-6 and 1e-4.

# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4), wd=.001)


# Within the first epoch, our error rate decreased.

# In[ ]:


learn.freeze()
learn.lr_find()
learn.recorder.plot()


# Saving the new model.

# In[ ]:


learn.save('model2_34')


# Confusion matrix to interpret the model's predictions.

# In[ ]:


learn.load('model1_34')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# Finding the most incorrect predictions.

# In[ ]:


interp.plot_top_losses(9, figsize=(10,10))


# Classifying an image.

# In[ ]:


img_baseball = open_image(Path(p_path)/'baseball_validation.jpg')
display(img_baseball)

pred_class, pred_idx, outputs = learn.predict(img_baseball)
pred_class


# > The issue is that the root folder 'sports-ball-images' is included as a class and therefore plays a strong roll in categorizing. I am working on fixing this.

# ### With very minimal fine-tuning, this image recognition model received an error rate of .18. To further improve my model, I would first need to add more images to my dataset because I believe the amount I have collected would not be enough to train classifiers sufficiently to distinguish such similar objects, and see how big of an effect that causes first.
