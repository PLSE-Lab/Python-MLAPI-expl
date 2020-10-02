#!/usr/bin/env python
# coding: utf-8

# # My First Rodeo with FastAI
# This is my first try with fastai so I'll keep it simple. Let me know where I could've done better.

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from fastai import vision
from fastai import metrics

import os
print(os.listdir("../input"))


# In[ ]:


train_imgs_path = '../input/train/train'
test_imgs_path = '../input/test/test'
labels_path = '../input/train.csv'
in_path = '../input/'


# In[ ]:


df = pd.read_csv(labels_path)
df['id'] = 'train/train/' + df['id']
df.head()


# In[ ]:


df.has_cactus.hist(grid=False, figsize=(5, 4), bins=np.arange(3)-0.3, width=0.6)
plt.xticks([0, 1])
plt.show()


# Number of images with cactus are ~3 times more than number of images without cactus. This might cause a bias towards presence of cactus. Let's see if transforms in fastai can handle this!

# In[ ]:


data = vision.ImageDataBunch.from_df(in_path, df, ds_tfms=vision.get_transforms(), size=224)
data = data.normalize(vision.imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(10, 8))


# In[ ]:


learn = vision.cnn_learner(data, vision.models.resnet34, metrics=metrics.accuracy)
learn.fit(2)


# `learn.save('stage-1')` throws error as `../input/models` is read only. Work around might be `torch.save(fastai.get_model(learn.model).state_dict(), 'stage-1')`. If you want to save optimizer as well then:
# 
# ```
# state = {'model': fastai.get_model(learn.model).state_dict(), 'opt':learn.opt.state_dict()}
# torch.save(state, 'stage-1')
# ```
# 
# We can certainly unfreeze and then fine-tune the model! But the model above seems preety good to me!
# 
# Let's interpret the results we got!

# In[ ]:


interp = vision.ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(5,5), dpi=60)


# Since there are only 2 classes, we certainly doesn't need to call `learn.most_confused(...)`.
# 
# Finally Predictions!

# In[ ]:


submission_df = pd.read_csv('../input/sample_submission.csv')
files = submission_df['id'].values
img_paths = ('../input/test/test/' + submission_df['id']).values


# In[ ]:


img_paths[:10]


# In[ ]:


from tqdm import tqdm
preds = []

for p in tqdm(img_paths):
    pred = learn.predict(vision.open_image(p))[-1].numpy()
    preds.append(pred)


# In[ ]:


submission_df['has_cactus'] = np.array(preds)[:, 1]
submission_df.head()


# In[ ]:


np.sum(np.array(preds)[:, 1] > 0.5), submission_df.shape


# In[ ]:


submission_df.to_csv('submission.csv', index=False)

