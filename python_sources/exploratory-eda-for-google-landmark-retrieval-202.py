#!/usr/bin/env python
# coding: utf-8

# Basic EDA for  Google Landmark Retrieval 2020. 
# Help taken from kernels
# 
# https://www.kaggle.com/huangxiaoquan/google-landmarks-v2-exploratory-data-analysis-eda
# https://www.kaggle.com/seriousran/google-landmark-retrieval-2020-eda
# 
# Please upvote if you like

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob
import cv2


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


train_file_path = '../input/landmark-retrieval-2020/train.csv'


# In[ ]:


df_train = pd.read_csv(train_file_path)


# In[ ]:


print("Training data size:", df_train.shape)
print("Training data columns:",df_train.columns)
print(df_train.info())


# In[ ]:


df_train.head(3)


# Data Sample

# In[ ]:


df_train.sample(3).sort_index()


# Data tail

# In[ ]:


df_train.tail(3)


# Explore the specific element

# In[ ]:


select = [4444, 10000, 14005]
df_train.iloc[select,:]


# Check if the data is None

# In[ ]:


print('data is None.')
missing = df_train.isnull().sum()
percent = missing/df_train.count()
missing_train_data = pd.concat([missing,percent],
                              axis=1, keys=['Missing','Percent'])
missing_train_data.head()


# Overall Basic Information

# In[ ]:


df_train['landmark_id'].describe()


# Object containing counts of unique values

# In[ ]:


sns.set()
print(df_train.nunique())
df_train['landmark_id'].value_counts().hist()


# Landmark_id distribuition

# In[ ]:


sns.set()
plt.title('Landmark_id Distribution')
sns.distplot(df_train['landmark_id'])


# In[ ]:


sns.set()
plt.title('Training set: number of images per class(line plot)')
sns.set_color_codes("pastel")
landmarks_fold = pd.DataFrame(df_train['landmark_id'].value_counts())
landmarks_fold.reset_index(inplace=True)
landmarks_fold.columns = ['landmark_id','count']
ax = landmarks_fold['count'].plot(logy=True, grid=True)
locs,labels = plt.xticks()
plt.setp(labels,rotation=30)
ax.set(xlabel="Landmarks",ylabel="Number of images")


# Training set: number of images per class(scatter plot)

# In[ ]:


sns.set()
landmarks_fold_sorted = pd.DataFrame(df_train['landmark_id'].value_counts())
landmarks_fold_sorted.reset_index(inplace=True)
landmarks_fold_sorted.columns = ['landmark_id','count']
landmarks_fold_sorted = landmarks_fold_sorted.sort_values('landmark_id')
ax = landmarks_fold_sorted.plot.scatter(     x='landmark_id',y='count',
     title='Training set: number of images per class(statter plot)')
locs, labels = plt.xticks()
plt.setp(labels, rotation=30)
ax.set(xlabel="Landmarks", ylabel="Number of images")


# Visualize outliers, min/max or quantiles of the landmarks count
# 

# In[ ]:


sns.set()
ax = landmarks_fold_sorted.boxplot(column='count')
ax.set_yscale('log')


# In[ ]:


sns.set()
res = stats.probplot(df_train['landmark_id'], plot=plt)


# Specific Basic Information

# In[ ]:


threshold = [2, 3, 5, 10, 20, 50, 100]
for num in threshold:    
    print("Number of classes under {}: {}/{} "
          .format(num, (df_train['landmark_id'].value_counts() < num).sum(), 
                  len(df_train['landmark_id'].unique()))
          )


# Most frequent landmark_ids

# In[ ]:


temp = pd.DataFrame(df_train.landmark_id.value_counts().head(10))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id', 'count']
temp


# In[ ]:


sns.set()
plt.title('Most frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp,
           label="count")
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.show()


# Least frequent landmark_ids

# In[ ]:


temp = pd.DataFrame(df_train.landmark_id.value_counts().tail(10))
temp.reset_index(inplace=True)
temp.columns = ['landmark_id', 'count']
temp


# In[ ]:


sns.set()
# plt.figure(figsize=(9, 8))
plt.title('Least frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=temp,
            label="Count")
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.show()


# **Test and index data**
# 
# 
# The query images are listed in the test/ folder, while the "index" images from which you are retrieving are listed in index/.
# 
# Each image has a unique id. Since there are a large number of images, each image is placed within three subfolders according to the first three characters of the image id (i.e. image abcdef.jpg is placed in a/b/c/abcdef.jpg).
# 
# 0-f in 0-f in 0-f

# In[ ]:


test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')
index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')


# Display examples
# 

# In[ ]:


plt.rcParams["axes.grid"] = False
f, axarr = plt.subplots(4, 3, figsize=(24, 22))

curr_row = 0
for i in range(12):
    example = cv2.imread(test_list[i])
    example = example[:,:,::-1]
    
    col = i%4
    axarr[col, curr_row].imshow(example)
    if col == 3:
        curr_row += 1


# In[ ]:




