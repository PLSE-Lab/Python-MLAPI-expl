#!/usr/bin/env python
# coding: utf-8

# # Google Landmark Retrieval 2020
# ![Image](https://i.ibb.co/McK51WV/thomas-kelley-ncy-Dc3s-CR-s-unsplash.jpg)
# 
# ## About this competition 
# Image retrieval is a fundamental problem in computer vision: given a query image, can you find similar images in a large database? This is especially important for query images containing landmarks, which accounts for a large portion of what people like to photograph.
# <br> 
# In this competition, the developed models are expected to retrieve relevant database images to a given query image (ie, the model should retrieve database images containing the same landmark as the query).
# <br>
# "Content-based" means that the search analyzes the contents of the image rather than the metadata such as keywords, tags, or descriptions associated with the image. The term "content" in this context might refer to colors, shapes, textures, or any other information that can be derived from the image itself. CBIR is desirable because searches that rely purely on metadata are dependent on annotation quality and completeness.
# <br>
# Having humans manually annotate images by entering keywords or metadata in a large database can be time consuming and may not capture the keywords desired to describe the image. The evaluation of the effectiveness of keyword image search is subjective and has not been well-defined. In the same regard, CBIR systems have similar challenges in defining success. "Keywords also limit the scope of queries to the set of predetermined criteria." and, "having been set up" are less reliable than using the content itself.
# <br><br>
# 
# ## About this Notebook
# In this kernel, I will briefly explain the structure of dataset. Then, I will visualize the dataset using Matplotlib and seaborn to gain as much insight as I can . Also I will approach this problem to provide a baseline solution and keep updating the kernel.
# 
# **This kernel is a work in Progress,and I will keep on updating it as the competition progresses and I learn more and more things about the data**
# 
#  <span style="color:red">If you find this kernel useful, Please Upvote it , it motivates me to write more Quality content</span>.

# ## Importing Necessary files

# In[ ]:


import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
import cv2
import numpy as np


# ## Reading the data

# In[ ]:


os.listdir('../input/landmark-retrieval-2020')


# In[ ]:


train_data = pd.read_csv('../input/landmark-retrieval-2020/train.csv')


# In[ ]:


train_data.info()


# In[ ]:


# Getting image paths
train_paths = glob.glob('../input/landmark-retrieval-2020/train/*/*/*/*')
test_paths = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')
index_paths = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')


# In[ ]:


print(f'[INFO] No. of Index Images : {len(index_paths)}')
print(f'[INFO] No. of Train Images : {len(train_paths)}')
print(f'[INFO] No. of Test Images : {len(test_paths)}')


# ## EDA

# In[ ]:


train_data.head()


# In[ ]:


train_data.describe()


# In[ ]:


train_data['landmark_id'].value_counts().hist()


# In[ ]:


# Landmark ID distribution
plt.figure(figsize = (10, 8))
plt.title('Category Distribuition')
sns.distplot(train_data['landmark_id'])
plt.show()


# In[ ]:


# Simple landmark id density plot
plt.title('Landmark id density plot')
sns.kdeplot(train_data['landmark_id'], color="tomato", shade=True)
plt.show()


# In[ ]:


# Landmark Id distribution and density plot 
plt.figure(figsize = (8, 8))
plt.title('Landmark id distribuition and density plot')
sns.distplot(train_data['landmark_id'],color='green', kde=True,bins=100)
plt.show()


# In[ ]:


# Cheking for null values if any
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)
missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()


# In[ ]:


# Occurance of landmark_id in decreasing order(Top categories)
temp = pd.DataFrame(train_data.landmark_id.value_counts().head(8))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']
temp


# In[ ]:


# Plot the most frequent landmark_ids
plt.figure(figsize = (9, 8))
plt.title('Most frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp,
            label="Count")
plt.show()


# In[ ]:


# Occurance of landmark_id in increasing order(Top categories)
temp = pd.DataFrame(train_data.landmark_id.value_counts().tail(8))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id','count']
temp


# In[ ]:


plt.figure(figsize = (9, 8))
plt.title('Least frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp,
            label="Count")
plt.show()


# In[ ]:


# uniques values in train 
train_data.nunique()


# ## Visualizing Images 

# In[ ]:


w = 10
h = 10
fig = plt.figure(figsize=(20, 20))
columns = 4
rows = 5

# prep (x,y) for extra plotting
xs = np.linspace(0, 2*np.pi, 60)  # from 0 to 2pi
ys = np.abs(np.sin(xs))           # absolute of sine

# ax enables access to manipulate each of subplots
ax = []

for i in range(columns*rows):
    example = cv2.imread(test_paths[i])
    img = example[:,:,::-1]
    # create subplot and append to ax
    ax.append( fig.add_subplot(rows, columns, i+1) )
    ax[-1].set_title("ax:"+str(i))  # set title
    plt.axis('off')
    plt.imshow(img)

# do extra plots on selected axes/subplots
# note: index starts with 0
ax[2].plot(xs, 3*ys)
ax[19].plot(ys**2, xs)

plt.show()  # finally, render the plot


# In[ ]:




