#!/usr/bin/env python
# coding: utf-8

# # Undertand the goal of this competition: 
# 
# From a set of medical images we need to give a diagnosis of hymorrage. To do that for each image we need to give a probability of the 5 subtypes of hymorrage if there is any.
# So the target column is Label whitch will contain the probability.
# 
# The 5 types of hymorrages:
# * 1_epidural
# * intraparenchymal
# * intraventricular
# * subarachnoid
# * subdural
# * any
# 
# In this stage 1 we have a crossover of patient IDs between test images and train images. We'll explore this specifity below.
# One of the weird rule of this competition is not use metadata of scan images to developp our model. So we will stick to the pixel array as much as possible but what the metadata hiding? :)
# 
# I hope this kernel will be helpful. Please give it an upvote if you like it

# 
# # Import libraries

# In[ ]:


import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import seaborn as sns
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import pydicom
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *


# # Load Data
# 
# We have 4 files. 
# 1. stage_1_train.csv : Csv with the columns ID and target
# 2. stage_1_train_images: contains images of the training set (dcm files) with the Id
# 3. stage_1_test_images: contains images of the test set (dcm files) with the Id
# 4. stage_1_sample_submission.csv: a sample submission file in the correct format. Contains Ids for the test set.
# 
# For the DICOM images files, we're going to use the pydicom library. 

# ### Load the csv files

# In[ ]:


train = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
sample_submission = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')


# Let's print the rows for the 2 first images

# In[ ]:


train.head(12)


# In[ ]:


train.shape


# In[ ]:


sns.countplot(train.Label)


# So, we have 674262 images( x6 = 445572)

# In[ ]:


train['filename'] = train['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".dcm")
train['Subtype'] = train['ID'].apply(lambda st: st.split('_')[2])
train['image_id'] = train['ID'].apply(lambda st:"ID_" + st.split('_')[1])
train.head(12)


# In[ ]:


len(train.image_id.unique())


# In[ ]:


train.groupby('Subtype').sum()


# Clearly the epidural cases are not sufficient

# I find it a little confusing to have all the images data in rows. We should have for each image 6 values of labels by hymorrages subtype. So let's pivot our dataframe.

# In[ ]:


train_df = train[['Label', 'image_id', 'Subtype']].drop_duplicates().pivot(index='image_id', columns='Subtype', values='Label').reset_index()
train_df.head(12)


# In[ ]:


train_df.shape


# In[ ]:


fig=plt.figure(figsize=(10, 8))
ncount = len(train_df['any'])
ax = sns.countplot(train_df['any'])
plt.title("Positive vs Negative results")
plt.xlabel("Number of Hymorrages")
ax2=ax.twinx()
#ax2.set_yticks(train_df['any'].count()*len(train_df)/100)
# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()
# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')
ax2.set_ylabel('Results [%]')

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.grid(None)


# In[ ]:


fig=plt.figure(figsize=(10, 8))
ncount = len(train_df['epidural'])
ax = sns.countplot(train_df['epidural'])
plt.title("Positive vs Negative epidural results")
plt.xlabel("Number of epidural Hymorrages")
ax2=ax.twinx()
#ax2.set_yticks(train_df['any'].count()*len(train_df)/100)
# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()
# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')
ax2.set_ylabel('Results [%]')

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.grid(None)


# In[ ]:


print(len(train_df[train_df['epidural']==1]))


# we have only 2761 cases of epidural hymorrage on 674258 images. So we're gonna overfit for sure.

# In[ ]:


fig=plt.figure(figsize=(10, 8))
Positive_df = train[train['Label']==1]
ncount = len(Positive_df['image_id'])

ax = sns.countplot(x="Subtype", hue="Label", data=Positive_df.loc[Positive_df['Subtype']!="any"])

plt.title("Hymorrages by type")

ax2=ax.twinx()
#ax2.set_yticks(train_df['any'].count()*len(train_df)/100)
# Switch so count axis is on right, frequency on left
ax2.yaxis.tick_left()
ax.yaxis.tick_right()
# Also switch the labels over
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')
ax2.set_ylabel('Results [%]')


for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
ax2.set_ylim(0,100)
ax.set_ylim(0,ncount)

ax2.grid(None)
ax.legend_.remove()


# ### Load the images files using Pydicom
# Pydicom library enables us to read a dicom image on python. So we have to store the directory paths on a list and so we can read them one by one using pydicom.read_file(imagename). I don't know for now if we can avoid the for loop. Let's see the first images

# In[ ]:


import pydicom
from pydicom.data import get_testdata_files


train_images_dir = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'
test_images_dir = '../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'

train_images_id = [f.split("/")[-1] for f in listdir(train_images_dir)]
test_images_id = [f.split("/")[-1] for f in listdir(test_images_dir)]

print(train_images_id[0:5])
print(test_images_id[0:5])

train_images = glob('../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/*.dcm')
print(train_images[0])
test_images = glob('../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/*.dcm')
print(test_images[0])


# In[ ]:


data_path = "../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/"

g = glob(data_path + '/*.dcm')

# Print out the first 5 file names to verify we're in the right folder.
print ("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
print ('\n'.join(g[:5]))


# In[ ]:


## Print the metadata of the first image

ds = pydicom.filereader.dcmread(train_images[12])
print(ds)


# In[ ]:


ds.pixel_array.shape


# In[ ]:


# plot the image using matplotlib
img = pydicom.read_file(train_images[0]).pixel_array
plt.imshow(img,cmap = plt.cm.bone) # colormap here is the colors of medical cm.bone
plt.show()


# In[ ]:


fig=plt.figure(figsize=(20, 20))
columns = 10; rows = 10
for i in range(1, columns*rows +1):
    ds = pydicom.dcmread(train_images[i])
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    fig.add_subplot


#     So here the size of heads is really confusing. Maybe there are some scans for kids or even babies. Maybe there are juste some scans that are empty. We can see only the section of the table for example image 15

# In[ ]:


img = pydicom.read_file(train_images[15]).pixel_array
plt.imshow(img,cmap = plt.cm.bone) # colormap here is the colors of medical cm.bone
plt.show()


# Let's visualize scans with hymorrage

# In[ ]:


train[train['Label']==1]


# In[ ]:


img_hymo_path = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_5c8b5d701.dcm'
img = pydicom.read_file(img_hymo_path).pixel_array
plt.imshow(img,cmap = plt.cm.bone) # colormap here is the colors of medical cm.bone
plt.show()


# I took the code from https://www.kaggle.com/allunia/rsna-ih-detection-eda-baseline to put the window and level manually

# In[ ]:


def set_manual_window(hu_image, custom_center, custom_width):
    min_value = custom_center - (custom_width/2)
    max_value = custom_center + (custom_width/2)
    hu_image[hu_image < min_value] = min_value
    hu_image[hu_image > max_value] = max_value
    return hu_image
def rescale_pixelarray(dataset):
    image = dataset.pixel_array
    rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept
    rescaled_image[rescaled_image < -1024] = -1024
    return rescaled_image


# In[ ]:


ds = pydicom.filereader.dcmread(img_hymo_path)
pixelarray = ds.pixel_array
plt.imshow(pixelarray, cmap=plt.cm.bone)
plt.grid(False)


# In[ ]:


rescaled_image = rescale_pixelarray(ds)
plt.imshow(rescaled_image, cmap=plt.cm.bone)
plt.grid(False)


# In[ ]:


org_windowed_image = set_manual_window(rescaled_image, 30, 80)
plt.imshow(org_windowed_image, cmap=plt.cm.bone)
plt.grid(False)

    


# The images in the train set and the test set are so bad we can't see anything and sometimes I'm not even able to see that its a section of a human brain. Why is that? Maybe we can improve the brightness but let's explore the data first

# In[ ]:


dataset = pydicom.dcmread(img_hymo_path)
#image = get_LUT_value(dataset.pixel_array, dataset.WindowWidth,
                             # dataset.WindowCenter)
array1 = dataset.pixel_array.copy()
array1[array1 < 30] = 0
array1[array1 > 70] = 0
plt.imshow(array1, cmap=plt.cm.bone)
plt.show()
plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)

farray = array1.flatten()
plt.figure(figsize=(10, 10))
plt.hist(farray,bins=100)
plt.xlim((0,500))
plt.ylim((0,2000))
plt.show()
    


# I'll be back with my analyse of dicom images :) 
