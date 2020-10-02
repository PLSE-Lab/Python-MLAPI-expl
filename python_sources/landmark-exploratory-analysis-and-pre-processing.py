#!/usr/bin/env python
# coding: utf-8

# I will be performing exploratory data analysis on the landmark training data in hopes of finding something useful for training a model later on. I have used this notebook as a starting reference: 
# 
# - https://www.kaggle.com/codename007/a-very-extensive-landmark-exploratory-analysis
# - https://github.com/jamesdietle/Kaggle2019/blob/master/GoogleLandmarkRecognition/GoogleLandmarkRecognition.ipynb

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_data = pd.read_csv('../input/landmark-retrieval-2020/train.csv')
print("Size of training data:", train_data.shape)
train_data.head()


# In[ ]:


print("Number of unique landmark ids:", len(train_data['landmark_id'].unique()))


# In[ ]:


# Looking for missing data

train_data.isnull().sum()


# In[ ]:


# Mostly recorded landmarks
temp = pd.DataFrame(train_data.landmark_id.value_counts().head(10))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']

# Plot the most frequent landmark_ids
plt.figure(figsize = (9, 8))
plt.title('Most frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp,
            label="Count")
plt.show()


# In[ ]:


# Class distribution
plt.figure(figsize = (10, 8))
plt.title('Category Distribuition')
sns.distplot(train_data['landmark_id'])

plt.show()


# In[ ]:


print("Number of classes under 20 occurences",(train_data['landmark_id'].value_counts() <= 20).sum(),'out of total number of categories',len(train_data['landmark_id'].unique()))


# # Image visualization

# Visualizing the images of the most observed landmarks id in the training data

# In[ ]:


def visualize_landmark(landmark_id=0):

    # Filtering the dataframe for most repeated landmark
    plotting_data = train_data[train_data['landmark_id']==train_data.landmark_id.value_counts().head(10).index[landmark_id]].reset_index()
    number_of_subplots = 3
    fig, axs = plt.subplots(number_of_subplots,figsize=(20,20))
    for i in range(number_of_subplots):

        # Creating file path
        image_name = plotting_data['id'][i] + '.jpg'
        first_initial, second_initial, third_initial = image_name[0]+'/', image_name[1]+'/', image_name[2]+'/'
        image_path = "../input/landmark-retrieval-2020/train/"+ first_initial + second_initial + third_initial + image_name

        # Displaying the image
        img = plt.imread(image_path)
        axs[i].imshow(img)


# In[ ]:


visualize_landmark(0)


# All of these seem to be random landmarks therefore they may be in huge numbers in the dataset.
# 

# Visualizing another landmark id that is lower on the top 10 list.

# In[ ]:


visualize_landmark(7)


# These landmarks look like castles.

# # Data Pre-processing

# There are 81313 unique landmark ids and many of them do not have a lot of images. It is better to falsely label these images rather than taking it into account in the assumption that the test set has these images in low numbers.

# In[ ]:


df = train_data.copy()


# In[ ]:


# How many unique landmarks are there?
groups=df.groupby('landmark_id')['id'].nunique()
groups.sort_values()


# In[ ]:


# # Make a sample of everything under 22 images
# under22=df.groupby('landmark_id')['id'].nunique()
# under22=under22.where(under22 < 22)
# under22=under22.dropna(how='any')
# under22=under22.index.tolist()

# # Change them into landmark id 99999999
# changed=df.replace([under22],99999999)

# # save this
# changed.to_csv("under22.csv", encoding='utf-8',index=False)


# ### Creating subset of top 20000 categories

# In[ ]:


# Get the top X categories
lst = df.groupby('landmark_id')['id'].nunique()

# Get the X largest categories
categories = 20000
top = lst.nlargest(categories)

# Create df subset
samplelocations = list(top.index.values)

#Receive files for subset
dfsubset=df[df['landmark_id'].isin(samplelocations)]

#Make Top 10
#dfsubset.to_csv("/home/jd/data/google/256/t10labels.csv", encoding='utf-8',index=False)
dfsubset.to_csv("t20000labels.csv", encoding='utf-8',index=False)

