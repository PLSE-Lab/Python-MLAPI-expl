#!/usr/bin/env python
# coding: utf-8

# # Model for Identifying the Time Period of Coins
# Artefacts and coins can be difficult to identify. While it would probably be unwise to rely only on a deep learning model to identify or date artefacts (even broadly speaking) I wanted to see how accurate such an algorithm could be.
# 
# ### Motivation
# 
# I'm still learning how to implement machine learning using the first lesson of the [fast.ai online course](https://course.fast.ai/).
# ## 1. Get the image data
# First, we need to download the images from the [Portable Antiquities Scheme](https://finds.org.uk/database/) (will be refered to as PAS). The method for downloading the images is as follows:
# 
# - If you are not already a member, register at the PAS website.
# - Perform a query for coins (or other artefacts if you decide to train a model on different objects)
# - Download the csv file for that query. (Note: If your query is too big, you may need to limit your query to specific time periods or dates.)
# - Repeat as many times as desired, attempting to get images of coins from each broad time period.

# In[ ]:


#Initialize packages.

import pandas as pd
import numpy as np
import requests
import shutil
import os.path
from os import path


# Now let's look at one of the csv files I used to create the dataset.

# In[ ]:


default = pd.read_csv('../input/coins-dataset/source_csv/coinlist.csv',encoding='ISO-8859-1')
df = default[['id','broadperiod', 'imagedir', 'filename']]
df.head(5)


# Now here's a breakdown of this data:
# 
# - **id**: The unique identifier for a given artefact or coin from the PAS database. We will append this to the end of our image names so that all files are unique.
# - **broadperiod**: The broad period of time from which the coin came from.
# - **imagedir**: The directory the image is located in on the website.
# - **filename**: The name of the file from the website.
# 
# Note that even with this information, some files weren't downloaded. More on this later.

# Using the data such as that from above, I was able to put together this function to download the image files. This specific function appends the "broadperiod" name to the front of the filename.

# In[ ]:


def download_img(broadperiod, imagedir, filename, idnum):
    # Sets the url and filename for the image. (Need to append the period name to filename for ML processing later.)
    # The parameters are converted to string objects in case they are initially numeric.
    image_url = "https://finds.org.uk/" + str(imagedir) + "medium/" + str(filename)
    filename = str(broadperiod) + '__' + str(idnum) + '.jpg'
    
    # Check if the file already exists before running requests.get. (Useful for rerunning batch_dl.)
    if path.exists('data/images/' + filename):
        print(filename + " already exists.")
    
    # When the file doesn't exist, downloads the file.
    else:
        r = requests.get(image_url, stream=True)
    
        # If the file is accessible via r, download it.
        if r.status_code == 200:
            r.raw.decode_content = True
        
            with open('data/images/' + filename, 'wb') as f:
                shutil.copyfileobj(r.raw,f)
        
            print('Downloaded: ', str(filename))
        else:
            print("Image couldn't be retreived")

            
    # This function iterates over a dataframe.
def batch_dl(df):
    i = 0
    while i < len(df):
        download_img(df.broadperiod[i], df.imagedir[i], df.filename[i], df.id[i])
        i += 1


# Since we want to download all the images from our existing csv files, we can just take all the data and concatenate them into one dataframe. Once we do this, we can run the batch_dl() function to download all the images we want.
# 
# ### Limitations
# Using this method, many images could not be downloaded from PAS. However, the sample was big enough to train a deep learning model.

# In[ ]:


gr = pd.read_csv('../input/coins-dataset/source_csv/greek_provincal.csv',encoding='ISO-8859-1')[['id','broadperiod', 'imagedir', 'filename']]
med = pd.read_csv('../input/coins-dataset/source_csv/medieval.csv',encoding='ISO-8859-1')[['id','broadperiod', 'imagedir', 'filename']]
pmed = pd.read_csv('../input/coins-dataset/source_csv/post_medieval.csv',encoding='ISO-8859-1')[['id','broadperiod', 'imagedir', 'filename']]
emed = pd.read_csv('../input/coins-dataset/source_csv/early_medieval.csv',encoding='ISO-8859-1')[['id','broadperiod', 'imagedir', 'filename']]

big = pd.concat([df, gr, med, pmed, emed]).drop_duplicates()
big.head()


# In[ ]:


# If you want to run this notebook off Kaggle, you can uncomment this to download the images.
# batch_dl(big)


# ## 2. Train the Model - Res 32
# Now is the part where we're going to try to train the model. Most of the code is derived from [this fast.ai lesson](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb).

# In[ ]:


# Import fastai modules
from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


# Get the file list for our image dataset.
fnames = get_image_files('../input/coins-dataset/images')
# Print part of the list.
fnames[:11]


# In[ ]:


# Set up regex for finding image filenames to find labels for image names.

pat = r'/([^/m]+?(?=__))'
imagepath = '../input/coins-dataset/images'


# In[ ]:


data = ImageDataBunch.from_name_re(imagepath, fnames, pat, ds_tfms=get_transforms(), size=224, bs=64).normalize(imagenet_stats)


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/kaggle/working/")


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# In[ ]:


data = ImageDataBunch.from_name_re(imagepath, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=32).normalize(imagenet_stats)


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1-50')


# ## Final Training
# Now we're going to fine-tune the model.

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-2))


# In[ ]:


learn.save('stage-2-50')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12))


# In[ ]:


interp.plot_top_losses(9, figsize=(15,15))


# It looks like there may be issues with how the coins are classified. The model tends to get confused between the varieties of medieval, post medieval, and early medieval coins. Perhaps these categories are variable based on a variety of factors, thus confounding the model.
# 
# Also, the images are being cropped in the processing, which also confounds the model.

# In[ ]:


learn.export('/kaggle/working/export.pkl')


# In[ ]:


defaults.device = torch.device('cpu')


# In[ ]:


img = open_image('../input/coins-dataset/images/EARLY MEDIEVAL__1000545.jpg')
img


# In[ ]:


learn = load_learner('/kaggle/working')


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class.obj

