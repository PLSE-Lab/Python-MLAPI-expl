#!/usr/bin/env python
# coding: utf-8

# This notebook is to analyse the labels and their relationships, without downloading the images.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import json
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# disable scientific notat
np.set_printoptions(suppress=True)


# In[3]:


#Utility method to read the data pickle
def read_dataset(filename):
    return pd.read_pickle(filename)


# We first start by creating the label to image mapping dataframe.
# The below method takes the data json and create a flat structure of `[Imageid, LabelId]`

# In[4]:


def create_label_pickle(JSONFILENAME, OUTPUTFILENAME):
    with open(JSONFILENAME) as data_file:
        data = json.load(data_file)
    dataset = []
    annotations = data['annotations']
    for annotation in annotations:
        labels = list(annotation['labelId'])
        for label in labels:
            a = {}
            a['imageId'] = annotation['imageId']
            a['labelId'] = label
            dataset.append(a)
    panda = pd.DataFrame(dataset)
    print(panda.head())
    panda.to_pickle(OUTPUTFILENAME)


# In[5]:


TRAIN_FILE = '../input/train.json'
TRAIN_LABEL_PICKLE = './train_label.pickle'

if os.path.isfile(TRAIN_LABEL_PICKLE) == False:
    create_label_pickle(TRAIN_FILE, TRAIN_LABEL_PICKLE)


# In[6]:


dataset = read_dataset(TRAIN_LABEL_PICKLE)
dataset = pd.DataFrame(dataset, dtype='int32')


# Now that we have the required structure we start with some basic analysis.
# 1. The number of distinct labels in the dataset.
# 2. The maximum labelId value [This will be used later].
# 3. The number of distinct images in the dataset.

# In[7]:


number_of_labels = dataset['labelId'].nunique() # Number of distinct labels
maximum_label_id = max(dataset['labelId']) # The maximum labelId value
number_of_images = dataset['imageId'].nunique() # Number of distinct images
print('Number of distinct labels in the dataset : ', number_of_labels)
print('Maximum id if labels in the dataset : ', maximum_label_id)
print('Number of distinct images in the dataset : ', number_of_images)


# One of the simplest analysis of the dataset can be done through counts. Let us start by creating counts of labels and images. This can easily be done using `pandas.groupby` method provided out of the box.

# In[8]:


# Count analysis for images
count_by_image_id = dataset.groupby('imageId')['imageId'].count().reset_index(name="count")
count_by_label_id = dataset.groupby('labelId')['labelId'].count().reset_index(name="count")


# Let us start by simple largest count.

# In[9]:


# Plot by label counts
print('Images with largest number of labels ')
count_by_image_id.nlargest(5, 'count')


# This tells us that maximum number of labels is < 25 . This is not much given we have 200+ labels. This laso gives us a rough idea how many labels we may have to predict for test images.
# 
# One other aspect of it to see how many images have high number of labels. We can get and ideas by plotting the counts in a bar graph.

# In[10]:


print(' Number of labels versus number of images with that many labels :')
a = count_by_image_id['count'].value_counts().sort_index().plot(kind = 'bar')


# This plot tells us that most images have between 3 to 9 labels. It is nice to get a near normal distribution as well :)
# Next we can do a similar analysis for labels also.

# In[11]:


# Plot by label counts
print('Labels associated with largest number of images ')
count_by_label_id.nlargest(5, 'count')


# Looks like label 66, 105 and 153 are the most occuring labels in the dataset. We can get a rough idea of all labels by plotting labelId with the count.

# In[12]:


count_by_label_id.plot(title='Labels versus how many times they occur')


# There are some labels are clearly occur more than the other. We can correlate them back to the actual images later and see what they could be.
# 
# One another aspect of labels is that they may occur together with other labels. It may tell us that they may be some how related. For example, ties may occur frequently with suits, shoes may occur often with socks etc. 
# 
# We can get this information by creating `NLabel * NLabel` matrix with zero values.
# Here NLabel is the maximum labelId.
# We increment `[i,j]` whenever `label i` and `label j` occur together. 
# 
# Disclaimer : The below piece is a slow and crude way to do this. If you find a more optimised way to do the same thing, mention in comments.

# In[ ]:


check_relation = np.zeros((maximum_label_id + 1 ,maximum_label_id + 1)) # adding one because labels are 1 indexed.
# we start by creating a dict with imageId as keys and list of labels as values.
relations = {}
for index, row in dataset.iterrows():
    imageId = row['imageId']
    labelId = row['labelId']
    if imageId in relations:
        # if this imageId is already there, map this label to all other labels already encountered.
        for l in relations[imageId]:
            check_relation[l][labelId] += 1
            check_relation[labelId][l] += 1
    else:
        # add this imageId to dict
        relations[imageId] = []
    # add this label to the imageId label's list
    relations[imageId].append(labelId)


# In[ ]:


# I am creating a clone here becasue in next few steps I am going to sort the matrix.
# I want to retain the original mappings also.
temp = np.copy(check_relation)


# In[ ]:


# Revert step in case I screw up later.
#check_relation = np.copy(temp)


# Let us see what we have in the relations. Let us take labelId 10 for example.

# In[ ]:


check_relation[10, :]


# We notice that label 10 occurs most frequently , 87 times, along-side label 66.
# 
# Let us use the `numpy.argsort` method to get the indices in sorted order.
# argsort does not actually sort the array but gives us the indices if the array were sorted.
# So for an array `[10 ,2 , 100, 4] ` it gives `[1, 3, 0, 2]`
# In our case this indices are just labelIds of other labels.

# In[ ]:


closest_companions = np.argsort(temp, axis=1) # axis = 1 to sort along the rows.
# Here I am filtering just the last three columns which have the highest values. 
# [:,::-1] is to reverse the three values because the values occur in ascending order 
# and I wanted them in descending
closest_companions = (closest_companions[:, -3:])[:,::-1]
print(closest_companions[10])


# I also wanted to check how many times a label occuring with its closest comapnion labels.
# For this we can simply sort the rows and take the last 3 values.

# In[ ]:


# the `temp` we created earlier will be used here.
sorted_closest_companions = temp
sorted_closest_companions.sort(axis = 1)
closest_companions_count = (sorted_closest_companions[:, -3:])[:,::-1]
print(closest_companions_count[10])


# The above two lists can together give us the closest companions for each label and how many times they occur together.
# 
# I will go ahead and make dataframe out of these arrays, for the most occuring companion.

# In[ ]:


companion = pd.DataFrame(columns=['companion'],data = closest_companions[1:,0], index = range(1, closest_companions.shape[0]))
companion['labelId'] = range(1, closest_companions.shape[0])
companion.head()


# In[ ]:


# I am taking closest_companions_count[1:,0] because
# the matrix indexing starts from 0 and we dont have a labelid = 1 in dataset.
# So no use keepin that value
companion_count = pd.DataFrame(dtype='int32',columns=['count'],data = closest_companions_count[1:,0], index = range(1, closest_companions_count.shape[0]))
companion_count['labelId'] = range(1, closest_companions_count.shape[0])
companion_count.head()


# I also wanted to check how often labels occur with their companions, with respect to the dataset.
# For example, if `label 1` occurs 100 times in the dataset, and it occurs with `label10` 80 times out of these 100, there may be something going on in there.
# 
# I started by plotting the labelId with theire counts in the dataset [something we did earlier too] and then overlaying the number if times they occur with their closest companion.

# In[ ]:


ax = companion_count.plot(title='Labels versus how many times they occur', x='labelId', y='count')
count_by_label_id.plot(ax= ax)


# The plot gives a general ideas that two trends are similar, but honestly, it looked weird to follow, so i tried doing it the old-school way.
# 
# I started by merging the two data frames we have, labelid versus count in the dataset, and label id versus number of time sit occured with its closest companion.

# In[ ]:


merged = pd.merge(companion_count, count_by_label_id, on='labelId')
merged.head()


# In[ ]:


# renaming columns to something more sensible.
merged.columns = ['companion_count', 'labelId', 'dataset_count']
merged.head()


# Just out of curiosity , lets calculate the percentage of times that happens.

# In[ ]:


merged['percentage'] = (merged['companion_count'] / merged['dataset_count'] ) * 100
merged.head()


# Lets also add the label if of the closest companion

# In[ ]:


merged = pd.merge(merged, companion, on='labelId')
merged.head()


# As a final touchup, lets reorder the columns.

# In[ ]:


merged = merged[['labelId', 'dataset_count', 'companion', 'companion_count', 'percentage']]
merged.head()


# You can read this as
# `<LabelId>` occurs in the dataset `<dataset_count>` times. Out of these `<dataset_count>` times, it occurs along side `<companion>` label , `<companion_count>` times, i.e, `<percentage>` of all occurances.

# In[ ]:


# Thanks for reading :) 


# In[ ]:




