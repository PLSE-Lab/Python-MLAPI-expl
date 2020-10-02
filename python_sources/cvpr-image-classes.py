#!/usr/bin/env python
# coding: utf-8

# ### Intro
# In this kernel I explore the dataset for the CVPR competition. We are given images from videos produced as a car drives around and records activity from the car's point of view. My primary purpose is to see what's in the images and get a general feel for the objects we are asked to segment. 
# 
# I'll also create a data frame of labels for the train set that can be used for more in-depth analysis. Here in this kernel we'll just look at counts and distributions for the train set.

# In[11]:


import os
import random
import numpy as np
import pandas as pd 
from skimage import io
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from dask.array.image import imread
from dask import bag, threaded
from dask.diagnostics import ProgressBar


# ### Files
# First let's quickly look at the default file structure. There are 3 directories and a sample submission.

# In[12]:


os.listdir('../input')


# Let's look first at the training images....

# In[13]:


def filecheck(dir):
    dir_size = 0
    filelist = os.listdir(dir)
    filelist.sort()
    print(dir)
    for i,name in enumerate(filelist):
        dir_size += os.path.getsize(os.path.join(dir, name))
    print("{:.1f} GB of {} files".format(dir_size/1024/1024/1024, i))
    print("showing sample files")
    print("\n".join(filelist[300:306]) + "\n")

dirs = ["../input/train_color","../input/train_label", "../input/test"]

for d in dirs[0:2]:
    filecheck(d)


# 92.3GB of image data for the train set images! The filenames are interesting. The prefixes of the files might represent dates the pictures were taken as in 170927 means year (20)17, month 9, and day 27.  The middle string looks like it represents times and/or frame numbers. You can see that each instance from the middle set of numbers is listed twice - once for Camera 5 and once for Camera 6. 
# 
# [Recovering the Videos](https://www.kaggle.com/andrewrib/recovering-the-videos) has a cool example of stringing together sequential frames into a video from one of the cameras.
# 
# Briefly looking at the test set files we see a more cyrptic naming convention. Our host recently uploaded a [Test Video List](https://www.kaggle.com/c/8899/download/test_video_list_and_name_mapping.zip) file that enables time sequencing of test images.

# In[14]:


j = os.listdir(dirs[2])
print("\n".join(j[0:6]))
print("{} files".format(len(j)))


# ### Images
# Jumping back to the training images - let's look at an image with labels.

# In[15]:


im = Image.open("../input/train_color/170908_061523257_Camera_5.jpg")
tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png")) // 1000
tlabel[tlabel != 0] = 255
# plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.4))
plt.imshow(im)
display(plt.show())


# OK, so there are some vehicles and such on the road, just as we would expect. To see what those things are, we can look into the png file and extract the values emedded inside. From the Data page....
# 
# #### The training images labels are encoded in a format mixing spatial and label/instance information:
# 
# * All the images are the same size (width, height) of the original images
# * Pixel values indicate both the label and the instance.
# * Each label could contain multiple object instances.
# * int(PixelValue / 1000) is the label (class of object)
# * PixelValue % 1000 is the instance id
# * For example, a pixel value of 33000 means it belongs to label 33 (a car), is instance #0, while the pixel value of 33001 means it also belongs to class 33 (a car) , and is instance #1. These represent two different cars in an image.
# 
# Clever, eh? Let's look.

# In[16]:


# cutting off everything after class 65, see note below
classdict = {0:'others', 1:'rover', 17:'sky', 33:'car', 34:'motorbicycle', 35:'bicycle', 36:'person', 37:'rider', 38:'truck', 39:'bus', 40:'tricycle', 49:'road', 50:'siderwalk', 65:'traffic_cone'}

tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_5_instanceIds.png"))
cls = np.unique(tlabel)//1000
unique, counts = np.unique(cls, return_counts=True)
d = dict(zip(unique, counts))
df = pd.DataFrame.from_dict(d, orient='index').transpose()
df.rename(columns=classdict, inplace=True)
df


# According to the data we have 5 cars, a bus, a tricycle and a traffic cone (see note below for 'traffic cone'). OK, sure... Let's also look at Camera 6 for the same instance.
# 

# In[17]:


im = Image.open("../input/train_color/170908_073302618_Camera_6.jpg")
tlabel = np.asarray(Image.open("../input/train_label/170908_073302618_Camera_6_instanceIds.png"))//1000
tlabel[tlabel != 0] = 255
plt.imshow(Image.blend(im, Image.fromarray(tlabel).convert('RGB'), alpha=0.6))

tlabel = np.asarray(Image.open("../input/train_label/170908_061523257_Camera_6_instanceIds.png"))
cls = np.unique(tlabel)//1000
unique, counts = np.unique(cls, return_counts=True)
d = dict(zip(unique, counts))
df = pd.DataFrame.from_dict(d, orient='index').transpose()
df.rename(columns=classdict, inplace=True)

display(plt.show())
df


# Camera 6 shows a different view as we might expect. The [First Look](https://www.kaggle.com/aakashnain/firstlook) kernel has more instances of images and masks.

# ### Labels (masks)
# Let's now pull labels for the training images and look at some basic stats. I'll write the labels out to csv for everyone's convenience.
# 
# Note: The code below can probably be further optimized for better performance. I have it down to less than 1/5 the time of the original code. The latest version uses Dask for multiprocessing. Coding with Dask seems easier than directly using the Multiprocessing package, but not faster, at least the way I'm doing it. 

# In[21]:


# dask imread version - in progress
def divmil():
    return 1
# labels is the image array
labels = imread("../input/train_label/*.png")
print(labels.shape, labels[0].shape, divmil)

# counts is the array of counts
    # look only at values between 33000 and 39999


# with ProgressBar():
#     labels.compute()


# In[ ]:


filenames = os.listdir(dirs[1])
# filenames = filenames[:100]            # for testing
fullpaths = ["../input/train_label/" + f for f in filenames]

# set up a bag
def get_ims(impath):
    tlabel = io.imread(impath, plugin='pil')
    cls = np.unique(tlabel)
    unique,counts = np.unique(cls//1000, return_counts=True)
    ds = dict(zip(unique, counts))
    return ds

labelbag = bag.from_sequence(fullpaths).map(get_ims)
with ProgressBar():
    labels = labelbag.compute()
    


# In[ ]:



labels_df = pd.DataFrame(labels, index=filenames, dtype='uint8') # dtype not working?
labels_df.fillna(value=0, inplace=True)
labels_df = labels_df.astype(int)
labels_df.rename(columns=classdict, inplace=True)      
labels_df.drop(columns=['others', 'rider', 'traffic_cone'], inplace=True)

labels_df.to_csv('train_labels.csv')
labels_df.head()


# We can look at the frequency of classes in the images by summing the occurrences across all images.

# In[ ]:



classes_df = pd.melt(labels_df)
groups = classes_df.groupby('variable')
sums = groups.sum()


sns.set(style='whitegrid')
ax = sns.barplot(x=sums.index, y=sums.value, color='steelblue')
ax.set(xlabel='', ylabel='count')
sns.despine(left=True)


# The most prevalent class is cars by far, as you might expect. As pointed out in [this discussion](https://www.kaggle.com/c/cvpr-2018-autonomous-driving/discussion/53845), only 7 of the classes will be used for evaluation. These are car, motorbicycle, bicycle, person, truck, bus, and tricycle. Class 65, traffic cone, is actually a false label, It comes from pixel value 65535 which represents the "ignoring label".
# 
# Anyway, let's look at the differences among images. Here are plots of Total Objects per image and Distinct Classes per image.

# In[ ]:


labels_df['objects'] = labels_df.sum(axis=1)
labels_df['classes'] = labels_df[labels_df>0].count(axis=1)-1
labels_df.clip(lower=0, inplace=True)   # crude fix for when no objects are seen
labels_df.head()


# In[ ]:


plt.figure();
plt.title("Total # of Objects")
labels_df['objects'].plot.hist()

plt.figure();
plt.title("# of Distinct Classes")
labels_df['classes'].value_counts().sort_index().plot.bar(color='steelblue')


# It's interesting to see quite a difference among the images, expecially for Total Counts.
# 
# There's a lot more that can be done with this data, of course, and I look forward to seeing some great kernels!
# 
# 
