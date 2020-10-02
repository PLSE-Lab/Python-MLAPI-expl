#!/usr/bin/env python
# coding: utf-8

# Can we use a model to predict, given an image, which institution it comes from ? 
# 
# I ask the question because, looking at the pictures, I saw very different looking shapes. I wonder if the two institutions have different way of taking the pictures (or maybe they render them differently ?)
# 
# If the pictures are so different from one institution to another,it could pertubate the model during training... 
# 
# Let's see if the pictures are so different...

# - Source of dataset: https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data/
# - Kernel: https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data/output

# # Install and import fastai2

# In[ ]:


get_ipython().system('pip install fastai2 > /dev/null')


# In[ ]:


import fastai2
from fastai2.vision.all import *


# # Get the data

# In[ ]:


path = Path('../input/prostate-cancer-grade-assessment')
path.ls()


# In[ ]:


df = pd.read_csv(path/'train.csv')
img_path = Path('../input/panda-train-png-images/train/')


# In[ ]:


df.head(3)


# # Create DataLoaders

# In[ ]:


# add .png to filenames
df['image_id'] = df['image_id'].apply(lambda x: str(x)+'.png')
df.head(3)


# In[ ]:


prostates = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   splitter=RandomSplitter(),
                   get_x=ColReader(0, pref=img_path),
                   get_y=ColReader(1),
                   item_tfms=Resize(224),
                   batch_tfms=aug_transforms()
                     )


# In[ ]:


dls = prostates.dataloaders(df, bs=16)
dls.show_batch()


# See they "grey" zone around the purple in "Radboud" pictures, whereas "Karolinska" doesn't seem to have that at all... Also, karolinska pictures seem to be nearly always straight vertical thin lines, whereas radboud look more like masses.
# 
# That's what got me started. Here are more examples:

# In[ ]:


dls.show_batch()


# # Model

# In[ ]:


learn = cnn_learner(dls, resnet50, metrics=accuracy)
learn.fit_one_cycle(1)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(k = 9)


# A single epoch gives us an accuracy of 100%. I suspect several possibilities
# 
# - the imagery process might differ
# - or they have similar process, but then don't encode the data similarly on the computer
# - it's not the same level of zoom
# - the preprocessing of <a href='https://www.kaggle.com/xhlulu/panda-resize-and-save-train-data/output'>this kernel</a> has somethings fishy 
# 
# In any case, finding a way for our model to look at the same thing when it sees images from different institution should probably a priority in this competition

# In[ ]:




