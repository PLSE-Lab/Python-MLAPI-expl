#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # 1. **Here we will first import all the libraries required**

# In[ ]:


#importing all the required libraries in this cell
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # 2. ** Now let us first read the dataset using pandas**

# In[ ]:


data = pd.read_csv("/kaggle/input/udemy-courses/udemy_courses.csv")
data.head(3)


# # 3. ** The different levels and domains in which the courses are offered**

# In[ ]:


print(f"Different levels of courses in the dataset are : {data['level'].unique()}")
print(f"Domains of the courses mentioned in the dataset are  : {data['subject'].unique()}")


# In[ ]:


data['is_paid']=data['is_paid'].astype(int)


# # 4. ** Let us analyze the data for each level**

# In[ ]:


level_wise_data = data.groupby(['level']).agg({'is_paid':'sum','num_subscribers':'sum','price':'mean','subject':'count'}).rename(columns={'subject':'Total Number of Courses in the level'})
level_wise_data


# In[ ]:


sns.barplot(level_wise_data.index,level_wise_data['Total Number of Courses in the level'],color = 'grey',label = 'Total number of courses')
sns.barplot(level_wise_data.index,level_wise_data['is_paid'],color = 'red',label = 'Number of paid courses')
plt.ylabel("")
plt.legend()


# In[ ]:


sns.barplot(level_wise_data.index,level_wise_data['num_subscribers'])
plt.legend()


# In[ ]:


sns.barplot(level_wise_data.index,level_wise_data['price'])
plt.ylabel("Average pricing of the courses in each level")
plt.legend()


# # 5.** Here we will explore the dataset using the various subjects/domains they are offered in **

# In[ ]:


subject_wise_data = data.groupby(['subject']).agg({'is_paid':'sum','num_subscribers':'sum','price':'mean','subject':'count'}).rename(columns={'subject':'Total Number of Courses in the level'})
subject_wise_data


# In[ ]:


ax=sns.barplot(subject_wise_data.index,subject_wise_data['Total Number of Courses in the level'],color = 'grey',label = 'Total number of courses')
ax=sns.barplot(subject_wise_data.index,subject_wise_data['is_paid'],color = 'red',label = 'Number of paid courses')
plt.xticks(rotation=15)
plt.ylabel("")
plt.legend()


# In[ ]:


sns.barplot(subject_wise_data.index,subject_wise_data['num_subscribers'])
plt.xticks(rotation=15)
plt.legend()


# In[ ]:


sns.barplot(subject_wise_data.index,subject_wise_data['price'])
plt.xticks(rotation=15)
plt.ylabel("Average pricing of the courses in each subject")
plt.legend()


# In[ ]:


print(f" The max and min prices are : {data['price'].max()},{data['price'].min()}")


# # 6.** In this step we analyze the data with respect to the various price brackets they are from(i.e., the cost of the course)**

# In[ ]:


bins = [-0.000000000000000001]
for i in range(21):
    bins.append(i*10)


# In[ ]:


category = pd.cut(data.price, bins)


# In[ ]:


data['price_bins']=category


# In[ ]:


data.head()


# In[ ]:


price_bins_data = data.groupby(['price_bins']).agg({"course_id":"count"}).rename(columns = {'course_id':'Number of courses in the price segment'})
price_bins_data.head()


# In[ ]:


plt.figure(figsize=(14,12))
sns.barplot(price_bins_data.index,price_bins_data['Number of courses in the price segment'].values,)
plt.ylabel('Number of courses in each price segment')
plt.xlabel('Price Segments')
plt.xticks(rotation = 90)
plt.show()


# In[ ]:





# In[ ]:




