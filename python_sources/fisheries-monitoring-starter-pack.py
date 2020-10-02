#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fastai.vision import *


# # Define paths

# In[ ]:


TRAIN = Path('../input/the-nature-conservancy-fisheries-monitoring/train/')
TEST = Path('../input/test-stg2/test_stg2/')
PATH = Path('../input/the-nature-conservancy-fisheries-monitoring/')

PATH_STR = '../input/the-nature-conservancy-fisheries-monitoring/train/'
PATH_WORKING = Path('/kaggle/working/')


# # Load data

# In[ ]:


os.listdir('../input/samplestg2/')


# In[ ]:


TRAIN.ls()


# In[ ]:


#TEST.ls()


# In[ ]:


sample_sub1 = pd.read_csv(PATH/'sample_submission_stg1.csv')
sample_sub2 = pd.read_csv('../input/samplestg2/sample_submission_stg2.csv')


# # Exploratory data analysis

# All files are in folders. So let's do some analysis of them
# 
# * **How many files per folder? Gives and idea of how long it will take to train**
# 
# 
# * **How many classes? Maybe there is some class imbalance and we need to train a model robust enough to deal with it with good performance.**
# 
# 
# * **File sizes? I'm still new to CNN so I can't tell right now how to deal with variable size input. Maybe on the next update.**

# ### Check number of files per folder

# In[ ]:


for idx,name in enumerate(os.listdir(TRAIN)):
    print(idx,name)


# So we have eight folders with images. There folders are the classes we want to predict:
# 
# **1. Other (meaning that there are fish present but not in the above categories)**
# 
# **2. SHARK (Shark, duh)**
# 
# **3. ALB (Albacore tuna)**
# 
# **4. LAG (Opah)**
# 
# **5. NoF (No Fish)**
# 
# **6. DOL (Mahi Mahi)**
# 
# **7. YFT (Yellowfin tuna)**
# 
# **8. BET (Bigeye tuna)**

# In[ ]:


num_files = [(name,len(os.listdir(TRAIN/name))) for idx,name in enumerate(os.listdir(TRAIN)) if name != ".DS_Store"]


# ### Barplot

# In[ ]:


files_df = pd.DataFrame(num_files,columns=['folder','count'])


# In[ ]:


files_df


# In[ ]:


files_df['count'].sum()


# In[ ]:


sns.barplot(x='folder',y='count',data=files_df)


# In[ ]:


os.listdir(TRAIN/'SHARK')[0]


# Ok, so we do have a **HUGE** class imbalance here, with most of the files consisting of Albacore tuna images. This might not be a problem in our first try, but we on the next updates we might want to take a deeper look.

# ### Check image size

# In[ ]:


from PIL import Image
from glob import glob
train_files = [(name,os.listdir(TRAIN/'{}'.format(name))) for idx,name in enumerate(os.listdir(TRAIN)) if name != ".DS_Store"]


# In[ ]:


# Define function to get image size
def get_sizes(files):
    '''Pass a list of file names and return image sizes'''
    sizes = []
    for folder in files:
        n = folder[0]
        f = folder[1]
        for files in f:
            image_file = PATH_STR+n+'/'+files
            with Image.open(image_file) as im:                
                sizes.append(im.size) # return value is a tuple, ex.: (1200, 800)
    return sizes


# In[ ]:


img_sizes = ['_'.join(map(str, s)) for s in get_sizes(train_files)]


# In[ ]:


sizes_df = pd.DataFrame(img_sizes,columns=['sizes'])
sizes_df.head()


# In[ ]:


c = sizes_df.reset_index().groupby('sizes')['index'].count().reset_index()
c


# In[ ]:


fig = plt.figure(figsize=(12,12))
ax = sns.barplot(x='sizes',y='index',data=c)
ax.set(xlabel='Sizes', ylabel='Count')


# We also have very different sizes of images, with the majority being 1280x720. What should we do? Well, on this first try we are going to use a smaller size of 512x512 just to see what happens.

# ### Take a look at some pictures - is everything alright?

# In[ ]:


sample_sharks = [PATH_STR+'SHARK'+'/'+files for files in train_files[0][1][0:10]]


# # Train model
# 
# Our files are in different folders so we need to access them. Luckly, fastai has a perfect API to do that called DataBlock. We can scrap each folder on the train set and get it's images, split into validation set and label them using just a few lines of code.

# ### Load data

# In[ ]:


src = ImageList.from_folder(path=TRAIN).split_by_rand_pct(0.2).label_from_folder().add_test_folder(TEST)


# ### Define transforms

# In[ ]:


tfms = get_transforms()


# ### Create databunch

# In[ ]:


data = src.transform(tfms,size=(670,1192)).databunch(bs=8).normalize(imagenet_stats)


# In[ ]:


print(data.classes,data.c)


# In[ ]:


data.show_batch(rows=3)


# The images have a lot of noise, such as water droplets on the camera lens, some were taken at night, fishes inside boxes etc. How can we tackle this? I don't know yet :) but maybe we could tweak the channels to highlight the most important thing: Fishes.
# 
# We could also do some image augmentation to zoom in and crop.

# ### Base model

# For some reason uusin Matthews Correlation 

# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=[accuracy],model_dir='/kaggle/working/')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(5,3e-3)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage1-fisheries')


# ### Fine tunning

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5,slice(3e-5,3e-4))


# In[ ]:


learn.save('stage2-fisheries')


# In[ ]:


learn.export('/kaggle/working/export.pkl')


# # Evaluation

# In[ ]:


sample_sub1.shape


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# Confusion matrix doesn't look so bad. We can see that even with huge class imbalance the model did quite a decent job. But we probably are going to do something about it. One strategy could be to randomly change some of the images of less frequent classes, save them and train the model using the modified images. 
# 
# Ex:
# 
# image_mod = transformation(image_original)
# 

# In[ ]:


interp.plot_top_losses(9, figsize=(7, 7))
# Maybe feed the network with top losses to improve score?


# # Make predictions and submit

# In[ ]:


WORKING_PATH = Path('/kaggle/working')


# In[ ]:


testset = ImageList.from_folder(TEST).add(ImageList.from_folder(PATH/'test_stg1'))


# In[ ]:


learn_preds = load_learner(WORKING_PATH, test=testset)


# In[ ]:


preds, _ = learn_preds.get_preds(ds_type=DatasetType.Test)


# In[ ]:


fish_classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


# In[ ]:


probabilities = preds[0].tolist()
[f"{index}: {probabilities[index]}" for index in range(len(probabilities))]
for idx,name in enumerate(fish_classes):
    print('{} : {}'.format(name,probabilities[idx]))
    


# In[ ]:


preds_df = pd.DataFrame(preds.numpy(),columns=fish_classes)


# In[ ]:


preds_df.head()
preds_df.shape


# In[ ]:


sample_sub2.head()


# In[ ]:


# remove file extension from filename
ImageId = list(sample_sub2['image'])


# In[ ]:


preds_df['image'] = ImageId


# In[ ]:


preds_df.head()


# In[ ]:


cols = list(preds_df.columns.values)
cols


# In[ ]:


preds_df = preds_df[['image','ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']]
preds_df.head()


# In[ ]:


preds_df.to_csv("/kaggle/working/submission.csv", index=False)


# In[ ]:




