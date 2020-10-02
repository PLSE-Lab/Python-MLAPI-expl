#!/usr/bin/env python
# coding: utf-8

# # APTOS 2019 Blindness Detection
# ### Detect diabetic retinopathy to stop blindness before it's too late 
# 
# 
# 

# Importing Dependencies and defining file paths. 

# In[ ]:


from fastai.vision import *
from fastai import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import glob
import torch


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')


# In[ ]:


PATH = Path('../input/aptos2019-blindness-detection')
train = PATH/'train_images'
test = PATH/'test_images'
train_folder = 'train_images'
model_dir = Path('/kaggle/working/')

train_df = pd.read_csv(os.path.join(PATH, 'train.csv'))
train_df['id_code'] = train_df['id_code'].apply(lambda x: f'{train_folder}/{x}.png')
train_df.head()


# In[ ]:


PATH.ls()


# In[ ]:


sns.countplot(train_df['diagnosis'])


# The plot above does indicate some imbalance between the classes. 

# In[ ]:


print(f"Size of Training set images: {len(list(train.glob('*.png')))}")
print(f"Size of Test set images: {len(list(test.glob('*.png')))}")


# #### On to Fastai.
# The Data block API makes it way easier to define a databunch. 
# I use a couple of transformations for the first stage of training.
# The values selected for these transformations are based on top performing solution summaries on the discussion tab. 

# In[ ]:


tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=180,
                      max_warp=0,max_zoom=1.25,p_lighting=0.4,
                      max_lighting=0.3,xtra_tfms=[flip_lr()] )


# Defining the data to be used for training and validation. 
# **Randomly setting aside 2% of the data for validation. **
# For the first training stage, i used padding as the resize method. 
# To get an understanding of how resize methods work in fastai check out the data augmentations part from fastai docs [here](https://docs.fast.ai/vision.transform.html#Data-augmentation)
# 

# In[ ]:


img_size = 456
bs = 64
src = (ImageList.from_df(train_df, PATH)
        .split_by_rand_pct()
        .label_from_df())

data = (src.transform(tfms, size=img_size, resize_method=ResizeMethod.PAD,padding_mode='zeros')
       .databunch(bs=32).normalize(imagenet_stats))

data

# The more simpler ImageDataBunch shortcut method. 
# data = ImageDataBunch.from_df(PATH, train_csv, folder='train_images', 
#                               suffix='.png', no_check=True, 
#                               ds_tfms=get_transforms(), size=512, bs=32).normalize(imagenet_stats)


# In[ ]:


data.train_ds


# In[ ]:


data.valid_ds


# Having a look at our Data. 

# In[ ]:


data.show_batch(rows=3, figsize=(10,8))


# # Training. 
# 
# As stated by the competition, they use the Quadratic Kappa score as the evaluation metric. 
# I also print the error_rate metric to get an idea how it performs side by side with Kappa. 
# 
# This is **Stage-1** Training. 
# > Fine tuning only the newly added final layers of the model whilst freezing the earlier layrers and using a pretrained(ImageNet) Resnet50.  

# In[ ]:


# Training
kappa = KappaScore()
kappa.weights = "quadratic"

learner = cnn_learner(data, models.resnet50, metrics=[error_rate, kappa])

learner.fit_one_cycle(4)


# Running the **LR finder** which would show us the optimal learning rate through a LR plot. 

# In[ ]:


learner.model_dir = '/kaggle/working'
learner.unfreeze()
learner.lr_find()


# In[ ]:


learner.recorder.plot()


# Based on the above plot and coupled with Jeremy Howards advices:
# - We select a value for the learning rate where the loss is minimum. 
# - Use this small learning rate to train the earlier layers. 
# 
# This time we fine tune the entire model, albiet with varying learning rates accross the layers. 
# - Smaller learning rate for earlier layers (slow learning as not much weight updation needed). 

# In[ ]:


learner.fit_one_cycle(4, max_lr=slice(1e-5, 1e-3))


# In[ ]:


learner.save('stage-2', return_path = True)


# This is **Stage-2** training. 
# > Here we create a new databunch with differnt sized pictures. We used 512x512 Images for stage-1 training. 
# > Now we use 456x456 to fine tune our model further. 
# > Following the same cycle, unfreezing --> lr_finder --> training with varying learning rates. 

# In[ ]:


tfms_456 = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,
                      max_warp=0,max_zoom=1.35,p_lighting=0.5,
                      max_lighting=0.2)

data_456 = (src.transform(tfms_456, size=512)
           ).databunch(bs=32).normalize(imagenet_stats)

data_456


# In[ ]:


learner_2 = cnn_learner(data_456, models.resnet50, metrics=[error_rate, kappa])

learner_2.load(model_dir/'stage-2')


# In[ ]:


learner_2.model_dir = '/kaggle/working'
learner_2.unfreeze()
learner_2.lr_find()


# In[ ]:


learner_2.recorder.plot()


# In[ ]:


learner_2.fit_one_cycle(4, max_lr=slice(1e-5, 1e-3))

