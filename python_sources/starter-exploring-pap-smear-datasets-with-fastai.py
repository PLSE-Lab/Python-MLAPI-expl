#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Let's use Fastai library to load the data, create our model and define the performance metrics.

# # Imports
# Let's import the required packages for our kernel

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from fastai import *
from fastai.vision import *
import pandas as pd
import matplotlib.image as mpimg


# # Exploring the dataset

# ## See where the data is (using terminal).

# In[ ]:


get_ipython().system('ls ../input/')


# As you can see, the datasets are divided into two subsets.
# * ``herlev_pap_smear`` refers to the Herlev pap smear dataset
# * ``sipakmed_fci_pap_smear`` referes to the SipakMed pap smear single-cell dataset

# ## Storing the paths of the datasets
# Let's store the paths of the respective datasets into variables

# In[ ]:


data_path = Path("../input/")
herlev_path = data_path/"herlev_pap_smear"
sipakmed_path = data_path/"sipakmed_fci_pap_smear"


# ## Exploring Herlev pap smear dataset
# ``herlev_path`` contains the path of the Herlev pap smear dataset. Let's look what's inside

# In[ ]:


herlev_path.ls()


# As you can see, there are total 7 classes in Herlev Pap Smear dataset. The dataset is structured as an Imagenet-style dataset. That's pretty convenient !!. Let's check the images inside...

# In[ ]:


class_paths = herlev_path.ls()
for c in class_paths:
    img_in_c_paths = get_image_files(c)
    print(f"Number of images in '{c.name}' : {len(img_in_c_paths)}")


# Let's look at some 10 images of a class, shall we...

# In[ ]:


sample_path = Path("../input/herlev_pap_smear/normal_superficiel")
imgs = sorted(get_image_files(sample_path))[:10]

for img_item in imgs:
    img = open_image(img_item)
    img.show()


# Wait, what's that?? There are segmentation masks as well !!!!. For each cell, there exists a segmentation mask named with a "-d" at the end of the filename. Let me show you..

# In[ ]:


sample_path = Path("../input/herlev_pap_smear/normal_superficiel")
imgs = sorted(get_image_files(sample_path))
imgs


# In[ ]:


mask, img = imgs[:2]
mask = open_image(mask)
img = open_image(img)
mask.show()
img.show()


# Alright, my intuition is correct. For data modelling, we'll ignore the mask images (as there is no codes file for each mask image).

# ## Exploring SipakMed dataset
# ``sipkamed_path`` contains the path of the Herlev pap smear dataset. Let's look what's inside

# In[ ]:


sipakmed_path.ls()


# As you can see, there are total 5 classes in SipakMed dataset. The dataset is structured as an Imagenet-style dataset. That's pretty convenient !!. Let's check the images of one class ...

# In[ ]:


classes = sipakmed_path.ls()
classes[0].ls()


# Oh my God...it's a mixup. I can see there are ``.dat`` files and ``.bmp`` files. Out of these, the ``.bmp`` files are the image files. Let's do some more digging...

# In[ ]:


one_class = classes[0]
one_class


# Let's explore the ``abnormal_Koilocytotic`` class

# In[ ]:


dat_and_img_files = sorted(one_class.ls())
dat_and_img_files


# We can see that for each image, there exists 2 ``.dat`` files. Let's check this intuition

# In[ ]:


sample_data = dat_and_img_files[:3]
sample_data


# In[ ]:


sample_img = sample_data[0]
open_image(sample_img)


# In[ ]:


nuc_data = pd.read_csv(sample_data[1], header=None)
nuc_data.head()


# Okay, looks like they are coordinates of pixels that surround the nuclues region of the cell image.
# The first column is the X coordinate and the second column is the Y coordinate.<br>What for the cytoplasm data?? Let's check that out...

# In[ ]:


cyto_data = pd.read_csv(sample_data[2], header=None)
cyto_data.head()


# Let's draw both of the repsective coordinates data on the cell image

# In[ ]:


img = mpimg.imread(sample_img)
plt.imshow(img)
plt.scatter(nuc_data.iloc[:, 0], nuc_data.iloc[:, 1], c="red")
plt.scatter(cyto_data.iloc[:, 0], cyto_data.iloc[:, 1], c="green")
plt.show()


# Looks cool. For starters, we will only be creating a data pipeline for classification. So, we'll ignore these ``.dat`` files.

# # Creating the data pipeline
# Let's use the data block API of the fastai library

# ## Herlev dataset pipeline

# In[ ]:


tfms = get_transforms(flip_vert=True, max_warp=0.0, max_zoom=0.)
herlev_data_block = (ImageList.from_folder(herlev_path)
                    .filter_by_func(lambda fname: "-d" not in fname.name)
                    .split_by_rand_pct(valid_pct=0.2, seed=0)
                    .label_from_func(lambda fname: "abnormal" if "abnormal" in fname.parent.name else "normal")
                    .transform(tfms, size=128)
                    .databunch(bs=16)
                    .normalize(imagenet_stats))


# We've built the data pipeline. Let's check that out...

# In[ ]:


herlev_data_block


# Woohoo...let's plot the data

# In[ ]:


herlev_data_block.show_batch(rows=4, figsize=(10 ,10))


# ## SipakMed dataset pipeline
# Let's built it for the SipakMed dataset

# In[ ]:


def labelling_func(fname):
    c = fname.parent.name
    if "abnormal" in c:
        return "abnormal"
    elif "benign" in c:
        return "abnormal"
    else:
        return "normal"

tfms = get_transforms(flip_vert=True, max_warp=0.0, max_zoom=0.9)

sipakmed_data_block = (ImageList.from_folder(sipakmed_path)
                      .split_by_rand_pct(valid_pct=0.2, seed=42)
                      .label_from_func(labelling_func)
                      .transform(tfms, size=128)
                      .databunch(bs=16)
                      .normalize(imagenet_stats))


# Let's check if the datablock was built correctly

# In[ ]:


sipakmed_data_block


# Yes !!!. Let's display some images of this datablock

# In[ ]:


sipakmed_data_block.show_batch(rows=4, figsize=(10, 10))


# ## TODO- Merging the datablock pipelines
# Let's merge both of these datablocks using the fastai library.

# # Conclusion
# This concludes our starter analysis. Let's see how creative you can get with my notebook for your own projects...
