#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastai.vision import *


# # Define paths

# In[ ]:


PATH = Path('../input/planet-understanding-the-amazon-from-space')
TRAIN = Path('../input/planet-understanding-the-amazon-from-space/train-jpg')
TEST = Path('../input/planet-understanding-the-amazon-from-space/test-jpg-v2')
PATH.ls()


# # Load data

# In[ ]:


df = pd.read_csv(PATH/'train_v2.csv')
samplesub = pd.read_csv(PATH/'sample_submission_v2.csv')


# In[ ]:


df.head()


# # Exploratory data analysis

#  ### Let's check how many files there are on training and test sets

# In[ ]:


print('Number of training files = {}'.format(len(df)))
print('Number of test files = {}'.format(len(samplesub)))

print('Number of training files = {}'.format(len(TRAIN.ls())))
print('Number of test files = {}'.format(len(TEST.ls())))


# ### Now to the classes. Since this is a multi-label classification each image can have N different labels, including:
# 
# 

# In[ ]:


labels = df.groupby('tags')['image_name'].count().reset_index()
labels.head()


# In[ ]:


labels.sort_values('image_name',ascending=False).head()


# In[ ]:


#sns.barplot(x=labels['tags'],y=labels['image_name'])
import matplotlib.ticker as ticker
plt.figure(figsize=(30,12))
ax = sns.barplot(x='tags',y='image_name',data=labels)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))


# We can see some labels are overrepresented on our dataset. For instance, clear primary 	dominates the dataset, followed by partly_cloudy primary. Let's take a look at some images of these classes.

# In[ ]:


sample_primary = df.loc[df['tags']=='clear primary'].head()
sample_partly_cloudy = df.loc[df['tags']=='partly_cloudy primary'].head()


# In[ ]:


sample_partly_cloudy


# In[ ]:


sample_primary


# In[ ]:


open_image(TRAIN/'train_2.jpg') # Clear primary


# In[ ]:


open_image(TRAIN/'train_17.jpg') # partly_cloudy primary


# # Train model

# We are going to train a naive model using Fastai v3

# ### Define transformations

# In[ ]:


#tfms = [[*rand_resize_crop(256),dihedral(),zoom(scale=1.05)],[]]
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# In[ ]:


src = ImageList.from_df(df,path=TRAIN,cols='image_name',suffix='.jpg').split_by_rand_pct(0.2).label_from_df(cols='tags',label_delim=' ')


# In[ ]:


data = src.transform(tfms).databunch(bs=64).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3)


# ### Training

# In[ ]:


arch = models.resnet50


# In[ ]:


learn = cnn_learner(data,arch,metrics=[fbeta],model_dir='/kaggle/working')


# In[ ]:


learn.lr_find()
# Find a good learning rate


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 1e-2


# In[ ]:


learn.fit_one_cycle(7,slice(lr))


# In[ ]:


learn.save('stage1-256-resnet50')


# ### Fine tuning

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(7,slice(1e-5,lr/5))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage2-256-resnet50')


# In[ ]:


learn.export('/kaggle/working/export.pkl')


# In[ ]:


test = ImageList.from_folder(TEST).add(ImageList.from_folder(PATH/'test-jpg-additional'))
len(test)


# In[ ]:


learn = load_learner(Path('/kaggle/working'), test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


thresh = 0.5
labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]


# In[ ]:


labelled_preds[:5]


# In[ ]:


fnames = [f.name[:-4] for f in learn.data.test_ds.items]


# In[ ]:


df_preds = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])


# In[ ]:


df_preds.to_csv('/kaggle/working/submission.csv', index=False)


# In[ ]:


a = df_preds.sort_values('image_name',ascending=True)
a.head()


# In[ ]:


df_preds.shape


# In[ ]:


samplesub.tail(50)


# In[ ]:


samplesub.shape


# In[ ]:


#! kaggle competitions submit planet-understanding-the-amazon-from-space -f {'/kaggle/working/submission.csv'} -m "My submission"


# In[ ]:




