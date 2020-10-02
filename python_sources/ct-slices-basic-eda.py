#!/usr/bin/env python
# coding: utf-8

# # Getting some basic idea of the CT-scans
# 
# The data contains CT-scans, which appear to be split into multiple slices. There must be some nice 3D software to combine them for the doctors. I am looking into the data in this kernel to get some idea what it is all about.
# 
# # A look into the lungs
# 
# The CT-scan image data is given as 2D layers of a 3D image. They are generally of the chest area, so showing the internals of the lungs (as far as I can tell..). 
# 
# In this kernel I just take a brief look at some of those images and their metadata. In my other [kernel](https://www.kaggle.com/donkeys/preprocessing-images-to-normalize-colors-and-sizes) I show how I preprocessed these. Here are two of those layered 3D images as animated gifs:
# 
# ![image1](https://i.imgur.com/KbL8C92.gif)
# 
# ![image2](https://i.imgur.com/TvPXWWg.gif)
# 
# One loops faster than the other since the scans have different numbers of frames.

# So, on with the show..

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pydicom
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pydicom
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image

tqdm.pandas()


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('ls /kaggle/input/osic-pulmonary-fibrosis-progression')


# In[ ]:


get_ipython().system('ls /kaggle/input/osic-pulmonary-fibrosispreprocessed/dataset')


# In[ ]:


df_train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
df_train.head()


# In[ ]:


df_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
df_test.head()


# In[ ]:


df_train_meta = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosispreprocessed/dataset/meta_files_train.csv")
df_train_meta.head()


# In[ ]:


df_test_meta = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosispreprocessed/dataset/meta_files_test.csv")
df_test_meta.head()


# ## Patient Smoking Statuses

# In[ ]:


df_train["SmokingStatus"].value_counts()


# ## Patient Genders

# In[ ]:


df_train["Sex"].value_counts()


# ## Patient Ages

# In[ ]:


df_train["Age"].value_counts().sort_index()


# ## Measurement counts by Patient

# In[ ]:


df_train["Patient"].value_counts()


# In[ ]:


df_train[df_train["Patient"] == "ID00229637202260254240583"]


# In[ ]:


get_ipython().system('ls /kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430 | wc -l')


# The above is interesting in showing that each patient has 6-10 measurement times in this dataset.

# ## DCM Files

# Data for a single patient appears to have multiple of the DCM files. It seems they are slices of the single CT-scan.

# In[ ]:


get_ipython().system('ls /kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00229637202260254240583/')


# ### Listing and Ordering the DCM files**

# In[ ]:


import glob

files = glob.glob("/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00229637202260254240583/*.dcm")
files


# In[ ]:


import re

files.sort(key=lambda f: int(re.sub('\D', '', f)))
files


# # Image Visuals

# First a single CT layer:

# In[ ]:


def show_img(img_path):
    ds = pydicom.dcmread(img_path)
    im = Image.fromarray(ds.pixel_array)
    #im = im.resize((DESIRED_SIZE,DESIRED_SIZE)) 
    #im.show()
    plt.imshow(im, cmap=plt.cm.bone)
    plt.show()
    

show_img(files[0])
#for file in files:
#    show_img(file)


# ## DCM Metadata
# 
# Each DCM file contains multiple metadata attributes:

# In[ ]:


import types

#ds = pydicom.dcmread('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00229637202260254240583/1.dcm')
ds = pydicom.dcmread('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00421637202311550012437/1.dcm')
attrs = dir(ds)
for attr in attrs:
    if attr.startswith("_"):
        continue
    if attr == "PixelData" or attr == "pixel_array": 
        continue
#            print(f"{attr}")
    var_type = type(getattr(ds,attr))
    if var_type == types.MethodType: 
        continue
        #    print(f"{attr}: {type(attr)}")
    #print(f"{attr}: {var_type}")
    print(f"{attr}: {getattr(ds,attr)}")
    


# ## Plotting CT-scan layers
# 
# If we plot all scan slices for a patient, in numerical order, and with associated coordinates, it shows the scan is sliced at regular intervals of 20 (whatever units it is..):

# In[ ]:


import math

def plot_images(images, cols=3):
#    plt.clf()
#    plt.figure(figsize(14,8))
    rows = math.ceil(len(images)/cols)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(14,5*rows))

    idx = 0
    for row in ax:
        for col in row:
            if idx > len(images)-1:
                break
            img_path= images[idx]
            ds = pydicom.dcmread(img_path)
            im = Image.fromarray(ds.pixel_array)
            col.imshow(im)
            col.title.set_text(ds.ImagePositionPatient)
            idx += 1
    plt.show()

plot_images(files)


# In the above plots, above each sub-plot are the "coordinates" of the slice inside the patient. The third number on each one is changing by 20 per layer. Is it the slice y-coordinate?

# ### Diffing Layers
# 
# Initially I thought the DCM file might be showing a timeframe, but it is actually just slices of the same as I show above. So, I tried to diff the layers in any case. The most interesting part is maybe that they appear to be circular slices, with some static noise surrounding the actual "beef" in each layer:

# In[ ]:


import math

def plot_image_diffs(images, cols=3):
#    plt.clf()
#    plt.figure(figsize(14,8))
    rows = math.ceil(len(images)/cols)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(14,5*rows))

    prev_img = None
    idx = 0
    for row in ax:
        for col in row:
            if idx > len(images)-1:
                break
            img_path= images[idx]
            ds = pydicom.dcmread(img_path)
            if prev_img is not None:
                print(f"no diff {idx}")
                diff = prev_img - ds.pixel_array
                #diff = ds.pixel_array - prev_img
                im = Image.fromarray(diff)
            else:
                print(f"diff {idx}")
                im = Image.fromarray(ds.pixel_array)
            col.imshow(im)
            col.title.set_text(ds.ImagePositionPatient)
            prev_img = ds.pixel_array
            idx += 1
    plt.show()

plot_image_diffs(files)


# # Preprocessed files
# 
# I have also uploaded a pre-processed dataset that reads all the .dcm files, takes out the medatadata for each, preprocessed each image to a matching color scaling and size, and writes them all out as .png files. Lets see here how that looks compared to the above. I will update this more later..

# In[ ]:


import glob

files = glob.glob("/kaggle/input/osic-pulmonary-fibrosispreprocessed/dataset/png/train/ID00229637202260254240583/*.png")
files


# In[ ]:


import re

files.sort(key=lambda f: int(re.sub('\D', '', f)))
files


# In[ ]:


def show_png(img_path):
    im = Image.open(img_path)
    plt.imshow(im, cmap=plt.cm.bone)
    plt.show()    

show_png(files[0])


# In[ ]:


import math

def plot_images(images, cols=3):
    rows = math.ceil(len(images)/cols)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(14,5*rows))

    idx = 0
    for row in ax:
        for col in row:
            if idx > len(images)-1:
                break
            img_path= images[idx]
            im = Image.open(img_path)
            col.imshow(im, cmap=plt.cm.bone)
            #col.title.set_text()
            idx += 1
    plt.show()

plot_images(files)


# In[ ]:


import math

def plot_image_diffs(images, cols=3):
    rows = math.ceil(len(images)/cols)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(14,5*rows))

    prev_img = None
    idx = 0
    for row in ax:
        for col in row:
            if idx > len(images)-1:
                break
            img_path= images[idx]
            img = Image.open(img_path)
            if prev_img is not None:
                print(f"no diff {idx}")
                img_data = np.asarray(img, dtype=np.uint8)
                diff = prev_img - img_data
                im = Image.fromarray(diff)
            else:
                print(f"diff {idx}")
                im = img
            col.imshow(im, cmap=plt.cm.bone)
            #col.title.set_text(ds.ImagePositionPatient)
            prev_img = np.asarray(img, dtype=np.uint8)
            idx += 1
    plt.show()

plot_image_diffs(files)


# In[ ]:


df_train_meta.describe()


# # Selected Pictures
# 
# As I looked through the images when implementing the pre-processing, I collected some cases that seemed to represent potentially useful augmentation functions for training and evaluation. 
# 

# In[ ]:


def show_train_png(img_id, idx):
    plt.figure(figsize=(12,8))
    im = Image.open(f"/kaggle/input/osic-pulmonary-fibrosispreprocessed/dataset/png/train/{img_id}/{idx}.png")
    plt.imshow(im, cmap=plt.cm.bone)
    plt.show()    


# ## Flat Shape 1
# 
# Many images are quite flat, with seemingly the head pointing upwards. The teeth are often showing as a white arch on the top.
# 
# Here is an example of teeth showing a bit:

# In[ ]:


show_train_png("ID00012637202177665765362", 1)


# Further along the layers, the same patient is a bit more rounded, I guess more in the middle of the chest:

# In[ ]:


show_train_png("ID00012637202177665765362", 7)


# ## Flat Shape 2
# 
# This one has the bones colored brighter white. Notice the upside down Y shape in the middle here. In the above figure the same structure seems to be upside down compared to this, which makes me think perhaps **vertical flip** could be a useful augmentation to experiment with?

# In[ ]:


show_train_png("ID00019637202178323708467", 2)


# In[ ]:


show_train_png("ID00019637202178323708467", 7)


# ## Round Shape
# 
# As opposed to the two flatter images above, some are more rounded:

# In[ ]:


show_train_png("ID00020637202178344345685", 1)


# ## Detached Parts
# 
# Sometimes the "parts" of the body show as quite detached in some parts of the layers. Guess it is another artifact of body shape/position, device properties etc. Several images show some form of this, but most none.
# 
# This one has both sides detached:

# In[ ]:


show_train_png("ID00047637202184938901501", 1)


# This one shows one part:

# In[ ]:


show_train_png("ID00122637202216437668965", 1)


# ## Ray Storm
# 
# This one looks a bit like there was a burst of energy going from right to left. Not sure how to augment for such effects.

# In[ ]:


show_train_png("ID00076637202199015035026", 1)


# ## Shiny Objects
# 
# Besides the above "ray storm", there were also multiple images with a shiny part emitting rays across. Here is one with such effect on top right corner.

# In[ ]:


show_train_png("ID00126637202218610655908", 1)


# ## Glowing Eyes
# 
# Not sure what this person had in their eyes.

# In[ ]:


show_train_png("ID00133637202223847701934", 1)


# ## Static
# 
# Some images had more static noise than others. So perhaps adding **noise** is a useful augmentation to try?

# In[ ]:


show_train_png("ID00123637202217151272140", 1)


# ## Teeth
# 
# I mentioned before how the teeth are sometimes very visible. This one was especially so:

# In[ ]:


show_train_png("ID00134637202223873059688", 1)


# ## Blurry
# 
# Some patients are blurred more than others. Again, perhaps adding **blur** could be another useful augmentation? For example, this one is a bit fuzzy:

# In[ ]:


show_train_png("ID00139637202231703564336", 1)


# And this one is much more so:

# In[ ]:


show_train_png("ID00264637202270643353440", 1)


# ## Tilt
# 
# The patients and devices are sometimes tilted and positioned in different ways. Some of this data is available in the image metadata, but rather than try to play with that, perhaps some level of **rotation** is another useful augmentation to try?

# In[ ]:


show_train_png("ID00196637202246668775836", 1)


# Overall, based on the above, I will play a bit with at least the following augmentations:
# 
# - brigthness levels: even looking at the above images after I tried to "normalize" their color levels, some are still brighter than others. 
# - Horizontal and vertical flip. Some images seem to show patients in different positions / angles both horizontally and vertically.
# - Adding noise. Some images had more static noise look than others. Not many very heavily, but some variation could be interesting to try.
# - Rotation levels. The angle of body changes in some images.
# - Dropping some parts of image. There is some variation in different parts being visible at different times.
# 
# How to best apply these to a 3D set of 2D images is another interesting question.. Or to model the 3D/2D aspect in general.
# 

# In[ ]:





# In[ ]:




