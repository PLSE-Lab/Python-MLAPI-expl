#!/usr/bin/env python
# coding: utf-8

# **About this challenge**
# 
# To assess the impact of climate change on Earth's flora and fauna, it is vital to quantify how human activities such as logging, mining, and agriculture are impacting our protected natural areas. Researchers in Mexico have created the VIGIA project, which aims to build a system for autonomous surveillance of protected areas. A first step in such an effort is the ability to recognize the vegetation inside the protected areas. In this competition, you are tasked with creation of an algorithm that can identify a specific type of cactus in aerial imagery.
# 
# In this kernel we will be trying to solve this challenge using CNN through **fast.ai library**
# 
# ![Fastailogo](https://images.ctfassets.net/orgovvkppcys/5EShj6ZsQFERrNd/af53baa732ce18025c51c9268ffd037b/image.png?w=648&q=100)

# **Loading necessary libraries**

# In[ ]:


from fastai.vision import *
from fastai import *
import os
import pandas as pd
import numpy as np
print(os.listdir("../input/"))


# In[ ]:


train_dir="../input/train/train"
test_dir="../input/test/test"
train = pd.read_csv('../input/train.csv')
test = pd.read_csv("../input/sample_submission.csv")
data_folder = Path("../input")


# **Analysing the given data**

# In[ ]:


train.head(5)


# In[ ]:


train.describe()


# **Getting the Data. **
# [reference](https://docs.fast.ai/vision.data.html)

# In[ ]:


test_img = ImageList.from_df(test, path=data_folder/'test', folder='test')
# Applying Data augmentation
trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
train_img = (ImageList.from_df(train, path=data_folder/'train', folder='train')
        .split_by_rand_pct(0.01)
        .label_from_df()
        .add_test(test_img)
        .transform(trfm, size=128)
        .databunch(path='.', bs=64, device= torch.device('cuda:0'))
        .normalize(imagenet_stats)
       )


# **Training the data using appropriate model. We have used [densenet](https://pytorch.org/docs/stable/torchvision/models.html) here**

# In[ ]:


learn = cnn_learner(train_img, models.densenet161, metrics=[error_rate, accuracy])


# **Finding the suitable learning rate**

# In[ ]:


learn.lr_find()


# **Plotting the Learning Rate**

# In[ ]:


learn.recorder.plot()


# **Now training the data based on suitable learning rate**

# In[ ]:


lr = 1e-02
learn.fit_one_cycle(10, slice(lr))


# In[ ]:


preds,_ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


test.has_cactus = preds.numpy()[:, 0]


# In[ ]:


test.to_csv('submission.csv', index=False)


# **References**
# * https://docs.fast.ai/
# * https://www.kaggle.com/kenseitrg/simple-fastai-exercise
# * https://www.kaggle.com/shahules/getting-started-with-cnn-and-vgg16
# 
