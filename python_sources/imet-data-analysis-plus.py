#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import cv2
import os


# ## Take a Glimpse

# In[28]:


train = pd.read_csv('../input/train.csv')
labels = pd.read_csv('../input/labels.csv')
test = pd.read_csv('../input/sample_submission.csv')

train.head(5)


# In[ ]:


print('Number of train samples: ', train.shape[0])
print('Number of test samples: ', test.shape[0])
print('Number of labels: ', labels.shape[0])


# ## Most Frequent and Infrequent Attributes

# In[ ]:


attribute_ids = train['attribute_ids'].values
attributes = []
for item_attributes in [x.split(' ') for x in attribute_ids]:
    for attribute in item_attributes:
        attributes.append(int(attribute))
        
att_pd = pd.DataFrame(attributes, columns=['attribute_id'])
att_pd = att_pd.merge(labels) #merge id with labels

frequent= att_pd['attribute_name'].value_counts()[:30].to_frame()

f, ax = plt.subplots(figsize=(12, 8))
ax = sns.barplot(y=frequent.index, x="attribute_name", data=frequent, palette="rocket", order=reversed(frequent.index))
ax.set_ylabel("Surface type")
ax.set_xlabel("Count")
sns.despine()
plt.title('Most frequent attributes')
plt.show()     


# In[ ]:


infrequent= att_pd['attribute_name'].value_counts(ascending=True)[:15].to_frame()
f, ax = plt.subplots(figsize=(12, 4))
ax = sns.barplot(y=infrequent.index, x="attribute_name", data=infrequent, palette="rocket", order=reversed(infrequent.index))
ax.set_ylabel("Surface type")
ax.set_xlabel("Count")
sns.despine() # remove the upper and right border
plt.title('Most infrequent attributes')
plt.show() 


# In[ ]:


train['Number of Tags'] = train['attribute_ids'].apply(lambda x: len(x.split(' ')))
f, ax = plt.subplots(figsize=(9, 8))
ax = sns.countplot(x="Number of Tags", data=train, palette="GnBu_d")
ax.set_ylabel("Surface type")
sns.despine()
plt.show()


# ## Analysis of Picture Size

# In[ ]:


width = []
height = []
for img_name in os.listdir("../input/train/")[-500:]:
    shape = cv2.imread("../input/train/%s" % img_name).shape
    height.append(shape[0])
    width.append(shape[1])
size = pd.DataFrame({'height':height, 'width':width})
sns.jointplot("height", "width", size, kind='reg')  
plt.show()


# In[29]:


size[:10]


# **So we can see that the length of one of two sides is 300**

# In[ ]:


print('The average height is ' + str(np.mean(size.height)))
print('The median height is ' + str(np.median(size.height)))
print('The average width is ' + str(np.mean(size.width)))
print('The median width is ' + str(np.median(size.width)))


# ## Attributes Statistics

# ### Pictures with Most Attributes

# In[ ]:


most_att = train[train['Number of Tags']>9]
least_att = train[train['Number of Tags']<2]

count = 1
plt.figure(figsize=[30,20])
for img_name in most_att['id'].values:
    img = cv2.imread("../input/train/%s.png" % img_name)
    plt.subplot(2, 3, count)
    plt.imshow(img)
    count += 1
plt.show


# ### Pictures with Least Attributes

# In[ ]:


count = 1
plt.figure(figsize=[30,20])
for img_name in least_att['id'].values[:6]:
    img = cv2.imread("../input/train/%s.png" % img_name)
    plt.subplot(2, 3, count)
    plt.imshow(img)
    count += 1
plt.show

