#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The project is used to label famous (and not-so-famous) landmarks in images.This Kernel explore the **train** and **test** datasets from [Google Landmark Recognition Challenge](https://www.kaggle.com/c/landmark-recognition-challenge). 
# 
# Please feel free to **fork and further develop** this Kernel. 
# 
# ![Google Landmark Challenge](https://lh3.googleusercontent.com/-3KCpDLA4tl0/V_xSEMwVIsI/AAAAAAAADXg/m6O0bgTtcDAPaXYn96U1x07E_gEvLuDGgCOcB/s1600/
# )
# 
# **Load Libraries**

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
from IPython.core.display import HTML 
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from urllib import request
from io import BytesIO
# io related
from skimage.io import imread
import os
from glob import glob
get_ipython().run_line_magic('matplotlib', 'inline')


# **Read Data **

# In[2]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Read the data
train_d = pd.read_csv("../input/train.csv")
test_d = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")


# **Inspect Data** 

# **Data shape**

# In[3]:


print("Train data size -  rows:",train_d.shape[0]," columns:", train_d.shape[1])
print("Test data size -  rows:",test_d.shape[0]," columns:", test_d.shape[1])
print("Submission data size -  rows:",submission.shape[0]," columns:", submission.shape[1])


# **Glimpse the data**

# Let's inspect the train and test sets

# In[4]:


train_d.head()


# Train set has three columns, first being an id for the image, the second being an url for the image and the third the id of the landmark associated with the image.

# In[5]:


test_d.head()


# Test set has two columns, first being an id for the image, the second being an url for the image. Let's see now the expected format for the submission file.

# In[6]:


submission.head()


# Submission has two columns, first being an id for the image, the second being the landmark. This has two elements: an landmark id that is associated with the image and its corresponding confidence score. Some query images may contain no landmarks. For these, one can submit no landmark id (and no confidence score).
# 
# **Data quality** : Let's look into more details to the data quality
# 
# **Train data quality** : Let's see if we do have missing values in the training set

# In[7]:


# missing data in training data set - missingt1 refers to missing values in train dataset
missingt1 = train_d.isnull().sum()
all_val = train_d.count()

missing_train_d = pd.concat([missingt1, all_val], axis=1, keys=['Missing', 'All'])
missing_train_d


# We see that we do not have any missing values (null values) in the training data
# 
# **Test data quality** : Let's see if we do have missing values in the test set

# In[8]:


# missing data in training data set - missingt2 refers to missing values in test dataset
missingt2 = test_d.isnull().sum()
all_val = test_d.count()

missing_test_d = pd.concat([missingt2, all_val], axis=1, keys=['Missing', 'All'])
missing_test_d


# We can see that we do not have any missing values (null values) in the test data
# 
# **Unique values** : Let's inspect the train and test data to check now many unique values are

# In[9]:


train_d.nunique()


# In the train dataset, there are only 14951 unique landmark_id data. All id's and url's are unique.Let's see now the test data to check now many unique values are

# In[10]:


test_d.nunique()


# All id's and url's are unique in the test data as well. Let's now check if we do have any id's or url's that are in both train and test set.

# In[11]:


# concatenate train and test datasets
concatenated = pd.concat([train_d, test_d])
# print the shape of the resulted data.frame
concatenated.shape


# In[12]:


concatenated.nunique()


# All id's and url's are unique for the concatenated data. That means we do not have any id's or url's from train dataset leaked in the test data set as well.
# 
# **Landmarks** : We already know how many distincts landmarks there are in the train set. Let's inspect now how many occurences are for these landscapes in the train set.

# In[13]:


plt.figure(figsize = (25,9))
plt.title('Landmark id density plot')
sns.kdeplot(train_d['landmark_id'], color="tomato", shade=True)
plt.show()


# Let's represent the same data as a density plot

# In[14]:


plt.figure(figsize = (25, 9))
plt.title('Landmark id distribuition and density plot')
sns.distplot(train_d['landmark_id'],color='blue', kde=True,bins=75)
plt.show()


# ** To ignore the warnings**

# In[15]:


import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")


# **To print Histogram**

# In[16]:


train_d['landmark_id'].value_counts().hist()


# Let's look now to the most frequent landmarks in the train set and also to the least frequent landmarks.
# # Occurance of landmark_id in decreasing order(Top categories)

# In[17]:


temp = pd.DataFrame(train_d.landmark_id.value_counts().head(25))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']
temp


# # Plot the most frequent landmark_ids count is 25

# In[18]:


plt.figure(figsize = (16, 16))
plt.title('Top 25 landmarks in train.csv data ')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp,
            label="Count")
plt.show()


# In[19]:


# Occurance of landmark_id in increasing order
temp1 = pd.DataFrame(train_d.landmark_id.value_counts().tail(25))
temp1.reset_index(inplace=True)
temp1.columns = ['landmark_id','count']
temp1


# # Plot the least frequent landmark_ids count is 25

# In[20]:


plt.figure(figsize = (16, 16))
plt.title('Last 25 landmarks in train.csv dataset')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp1,
            label="Count")
plt.show()


# #Class distribution

# In[21]:


plt.figure(figsize = (16, 16))
plt.title('Category Distribuition')
sns.distplot(train_d['landmark_id'])

plt.show()


# **Image Thumbnails **
# Let's inspect also the images. We create a function to display a certain number of images, giving a list of images urls. We show here a number of `50` images of the `Pantheon` in Rome, which is the 5th ranged landmark in the selection of landmarks, based on number of occurences. We will define two functions to display landmarks.

# In[22]:


def displayLandmarkImages(urls):
    
    imageStyle = "height: 60px; margin: 2px; float: left; border: 1px solid blue;"
    imagesList = ''.join([f"<img style='{imageStyle}' src='{u}' />" for _, u in urls.iteritems()])

    display(HTML(imagesList))
    
    
def displayLandmarkImagesLarge(urls):
    
    imageStyle = "height: 100px; margin: 2px; float: left; border: 1px solid blue;"
    imagesList = ''.join([f"<img style='{imageStyle}' src='{u}' />" for _, u in urls.iteritems()])

    display(HTML(imagesList))


# In[23]:


IMAGES_NUMBER = 50
landmarkId = train_d['landmark_id'].value_counts().keys()[9]
urls = train_d[train_d['landmark_id'] == landmarkId]['url'].head(IMAGES_NUMBER)
displayLandmarkImages(urls)


# Let's visualize now 5 images for each of the first 8 landmarks, ordered by the number of occurences.

# In[24]:


LANDMARK_NUMBER = 5
IMAGES_NUMBER = 8
landMarkIDs = pd.Series(train_d['landmark_id'].value_counts().keys())[1:LANDMARK_NUMBER+1]
for landMarkID in landMarkIDs:
    url = train_d[train_d['landmark_id'] == landMarkID]['url'].head(IMAGES_NUMBER)
    displayLandmarkImagesLarge(url)


# If we change key value in Keys() we will get different images every time in dataset 

# In[25]:


from IPython.display import Image
from IPython.core.display import HTML 

def display_category(urls, category_name):
    img_style = "width: 180px; margin: 0px; float: right; border: 1px solid black;"
    images_list = ''.join([f"<img style='{img_style}' src='{u}' />" for _, u in urls.head(25).iteritems()])
    display(HTML(images_list))

category = train_d['landmark_id'].value_counts().keys()[2018]
urls = train_d[train_d['landmark_id'] == category]['url']
display_category(urls, "")


# **Baseline Submission** : We are using a random guess, normalized by the frequency in the training set to prepare a submission file.The solution is picked up from Kevin Mader's Kernel, `Baseline Landmark Model`.

# In[26]:


# take the most frequent label
freq_label = train_d['landmark_id'].value_counts()/train_d['landmark_id'].value_counts().sum()


# In[27]:


# submit the most freq label
submission['landmarks'] = '%d %2.2f' % (freq_label.index[0], freq_label.values[0])
submission.to_csv('submission.csv', index=False)


# In[28]:


np.random.seed(2018)
r_idx = lambda : np.random.choice(freq_label.index, p = freq_label.values)


# In[29]:


r_score = lambda idx: '%d %2.4f' % (freq_label.index[idx], freq_label.values[idx])
submission['landmarks'] = submission.id.map(lambda _: r_score(r_idx()))
submission.to_csv('rand_submission.csv', index=False)


# In[30]:


# Now Lets extract the website name and see their occurances

# Extract site_names for train data
temp_l1 = list()
for path in train_d['url']:
    temp_l1.append((path.split('//', 1)[1]).split('/', 1)[0])
train_d['site_name'] = temp_l1

# Extract site_names for test data
temp_l1 = list()
for path in test_d['url']:
    temp_l1.append((path.split('//', 1)[1]).split('/', 1)[0])
test_d['site_name'] = temp_l1

#We have added one new column "site_name".lets see
print("Training data size",train_d.shape)
print("test data size",test_d.shape)


# In[31]:


# New columns added to existing dataset
train_d.head()


# In[32]:


#New column in existing test data 
test_d.head()


# In[33]:


#In this we are creating a duplicate table to drop column url in train dataset
train_d1 = train_d
train_d1.head()


# In[34]:


# url column is dropped in train dataset
train_d1 = train_d1.drop('url',1) 
train_d1.head()


# In[35]:


# Occurance of site in decreasing order(Top categories) in train dataset
temp = pd.DataFrame(train_d1.site_name.value_counts())
temp.reset_index(inplace=True)
temp.columns = ['site_name','count']
temp


# In[36]:


#As we can see there are total 16 unique sites.
# Plot the Sites with their count
plt.figure(figsize = (16, 16))
plt.title('Sites with their count')
sns.set_color_codes("pastel")
sns.barplot(x="site_name", y="count", data=temp,
            label="Count")
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()


# In[37]:


#In this we are creating a duplicate table to drop column url in test dataset
test_d1 = test_d
test_d1.head()


# In[38]:


# url column is dropped in train dataset
test_d1 = test_d1.drop('url',1)
test_d1.head()


# In[39]:


#occurances of sites in test_data
# Occurance of site in decreasing order(Top categories)
temp = pd.DataFrame(test_d.site_name.value_counts())
temp.reset_index(inplace=True)
temp.columns = ['site_name','count']
temp


# In[40]:


#Total unique sites are 25 in test data and some are different from train_data
# Plot the Sites with their count
plt.figure(figsize = (16, 16))
plt.title('Sites with their count')
sns.set_color_codes("pastel")
sns.barplot(x="site_name", y="count", data=temp,
            label="Count")
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()


# In[41]:


train_d2 = train_d # Dataset with site_name column
test_d2 = test_d # Dataset with site_name column
train_d2.head()


# In[42]:


test_d2.head()


# In[43]:


# To drop site_name column in train dataset this the original dataset for reference
train_d2 = train_d2.drop('site_name',1)
train_d2.head()


# In[44]:


# To drop site_name column in test dataset this the original dataset for reference
test_d2 = test_d2.drop('site_name',1)
test_d2.head()


# **Random Guessing**

# In[54]:


import pylab as pl
pl.seed = 0
N = 1500
probs = train_d.landmark_id.value_counts() / train_d.shape[0]
probs = probs.iloc[:N]
probs = pd.DataFrame({'landmark_id': probs.index,
                      'probability': probs.values}, index=pl.arange(N))
T = pd.merge(train_d, probs, on='landmark_id', how='outer')
inx = pl.randint(0, T.shape[0], submission.shape[0])
submission['landmark_id'] = T.landmark_id.iloc[inx].values
submission['prob'] = T.probability.iloc[inx].values
submission['landmarks'] = submission.landmark_id.astype(str) + ' ' + submission.prob.astype(str)
submission[['id','landmarks']].head()
submission[['id','landmarks']].to_csv('submission_inner.csv', index=False)


# In[50]:


import pylab as pl
pl.seed = 0
N = 1500
probs = train_d.landmark_id.value_counts() / train_d.shape[0]
probs = probs.iloc[:N]
probs = pd.DataFrame({'landmark_id': probs.index,
                      'probability': probs.values}, index=pl.arange(N))
T = pd.merge(train_d, probs, on='landmark_id', how='inner')
inx = pl.randint(0, T.shape[0], submission.shape[0])
submission['landmark_id'] = T.landmark_id.iloc[inx].values
submission['prob'] = T.probability.iloc[inx].values
submission['landmarks'] = submission.landmark_id.astype(str) + ' ' + submission.prob.astype(str)
submission[['id','landmarks']].head()
submission[['id','landmarks']].to_csv('submission_outer.csv', index=False)


# In[52]:





# **Feedback requested **
# 
# Your suggestions and comments for improvement of this Kernel are much appreciated. And, of course, if you like it,** upvote!**
