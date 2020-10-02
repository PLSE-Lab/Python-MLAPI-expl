#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math


# In[ ]:


#our dataset
df=pd.read_table('../input/fruits-with-colors-dataset/fruit_data_with_colors.txt')


# In[ ]:


df.head()


# In[ ]:


type(df)


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.head()


# In[ ]:


labels = dict(zip(df.fruit_label.unique(), df.fruit_name.unique()))   


# In[ ]:


df['fruit_name'].value_counts()


# **Steps for KNN**
# 1. Take a random record and remove from data.
# 2. For each record in data: compute the distance between xq and the point,store the distance 
# 3. Take the 3 smallest distances bcoz k=3
# 4. Take majority out of 3

# In[ ]:


#taking random record and storing in xq
xq = df.sample()


# In[ ]:


# droping the xq from data using index value
df.drop(xq.index, inplace=True)


# In[ ]:


df.shape


# In[ ]:


xq_final = pd.DataFrame(xq[['mass', 'width', 'height', 'color_score']])


# In[ ]:


xq_final


# In[ ]:


# calculating ecludian distance
def cal_distance(x):      
    a = x.to_numpy()
    b = xq_final.to_numpy()    
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b[0])]))
    return distance


# In[ ]:


# calculating distance
df['distance'] = df[['mass', 'width', 'height', 'color_score']].apply(cal_distance, axis=1)


# In[ ]:


#sorting the values based on distance
df_sort = df.sort_values('distance',ascending=True)


# In[ ]:


# taking top 3 records because k is 3
df_after_sort = df_sort.head(3)


# In[ ]:


df_after_sort.reset_index()


# In[ ]:


df_after_sort.iloc[0]


# In[ ]:


count = [0 for i in range(0, len(df['fruit_label'].unique()))]
for xi in range(0, len(df_after_sort)):       
    if df_after_sort.iloc[xi]['fruit_label'] == 1:        
        count[0] = count[0]+1
    elif df_after_sort.iloc[xi]['fruit_label'] == 2:        
        count[1] = count[1]+1
    elif df_after_sort.iloc[xi]['fruit_label'] == 3:        
        count[2] = count[2]+1
    elif df_after_sort.iloc[xi]['fruit_label'] == 4:        
        count[3] = count[3]+1


# In[ ]:


def max_num_in_list_label(list):
    maxpos = list.index(max(list)) +1
    return labels[maxpos]


# In[ ]:


#getting the label and verifying with the class label in xq
if max_num_in_list_label(count) in xq.values:
    print("success")
else:
    print("not success")


# In[ ]:





# In[ ]:





# In[ ]:




