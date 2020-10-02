#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Reading the csv file
unv_df=pd.read_csv("/kaggle/input/world-university-rankings/cwurData.csv")


# In[ ]:


#displaying the top 10 data
unv_df.head(10)


# In[ ]:


#Calculate the no of rows and columns
unv_df.shape


# In[ ]:


unv_df.dtypes


# From the above it is clear that the institution and country are categorical data. Rest of the data are numeric with data type as integer and float respectively.

# In[ ]:


#Check for the missing values 

unv_df.isna().any()


# In[ ]:


#Calculating the no of missing values for broad_impact column
unv_df['broad_impact'].isna().sum()


# There are 200 missing values in the broad_impact column.Broad_impact values are not available for year 2014 and 2015. Also the mean and median are equal which means the data is normally distributed. 

# In[ ]:


unv_df.nunique()


# In[ ]:


unv_df["year"].value_counts()


# 
# **All the rankings are based from the year 2012-2015. We will divide our data into four parts for each year respectively.**

# **1. Analysis of the data for year 2012**

# In[ ]:


unv_df_2012=unv_df.loc[unv_df['year'] == 2012]
unv_df_2012.head(10)


# In[ ]:


#Shape of the data

unv_df_2012.shape


# In[ ]:


#Missing values in the columns

unv_df_2012.isna().any()


# In[ ]:


#Count the missing values in the broad_impact data set

unv_df_2012.isna().sum()


# In[ ]:


#Drop the broad_impact column as the entire data is missing for all the rows 

unv_df_2012.drop("broad_impact",axis=1,inplace=True)


# The above shown warning can be removed by the following code :-
# 
#         unv_df_2012_new = unv_df_2012.drop('Item', axis=1).
#         
#  But I am ignoring it to avoid so many names for the same set of the data       
#                               
#         
# 

# In[ ]:


#List of countries in the dataset

unv_df_2012["country"].unique()


# In[ ]:


unv_df_2012.describe().T


# In[ ]:


sns.pairplot(unv_df_2012)


# **Observations**
# 
# 1. World Rank is from 1 to 100 for all the universities.
# 2. Most of the universities are top or second best universities respectively from there country.
# 3. Most of the univesities rank higher in terms of quality of education,alumni employemnt,quality of         faculity,publication,influence,citations and patents (rank=101). Here Mean>Median. the graph is right skewed.
# 4. The world Rank and the national rank is directly proportional to each other . As the national rank rises , the world rank rises too.But in some of the cases the national rank is less but the world rank is much more.
# 5. The world rank and the quality of education is evenly distributed for most of the data. The universities with rank 101 in terms of quality of education has world rank between 25 to 100.
# 6. The world rank and the alumni employment is evenly distributed for most of the data. The universities with rank 101 in terms of alumni employment has world rank between  15 to 100.
# 7. The world rank and the quality of faculty is evenly distributed for most of the data. The universities with rank 101 in terms of quality of faculty has world rank ranges between 30 to 100.
# 8. The world rank and the publications is evenly distributed for most of the data. The universities with rank 101 in terms of publications doesn't have world rank in top 20.
# 9. The word rank and the influence is evenly distributed for most of the data.The universities with rank 101 in terms of influence doesn't have world rank in top 40.
# 10. The world rank and the citations  is evenly distributed for most of the data. The universities with rank 101 in terms of citations doesn't have world rank in top 20.
# 11. The world rank and the patents is evenly distributed for most of the data. The universities with rank 101 in terms of patents doesn't have the world rank in top 5.
# 12. The world rank and the score is indirectly proportional to each other. The universities with low score have comparatively higher rank.
# 
# 

# In[ ]:


# Correlation with heat map
import matplotlib.pyplot as plt
import seaborn as sns
corr =unv_df_2012.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
# create a mask so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)


# **Observation**
# 
# 1. World rank and Score are highly corelated to each other.
# 2. Publications and quality of education is least corelated to each other.

# In[ ]:


#Top ten universities in terms of world rank 

unv_df_2012_new= unv_df_2012.head(10)
plt.scatter(unv_df_2012_new.country,unv_df_2012_new.institution)
plt.xlabel('countries')
plt.ylabel('universities')
plt.title('rank')
plt.show()


# In[ ]:


#Top ten universities of the USA as per the national rankings

unv_df_2012_USA=unv_df_2012.loc[unv_df_2012['country']=="USA"]

unv_df_2012_USA.sort_values('national_rank')
print(unv_df_2012_USA["institution"].head(10))


# In[ ]:


# University of USA world ranking 
plt.hist(unv_df_2012_USA.world_rank,bins=2) #histogram
plt.title('Universities of USA')
plt.xlabel("rank of institutions")
plt.show()


# **Most of the US universities are ranked between 1 to 50. **

# In[ ]:


#Alumni employed by the top ten universities  

plt.figure(figsize=(25,7)) # this creates a figure 20 inch wide, 7 inch high
sns.barplot(unv_df_2012_new['institution'], unv_df_2012_new['alumni_employment'])


# In[ ]:


#Impact of score to the 
plt.figure(figsize=(25,7)) # this creates a figure 20 inch wide, 7 inch high
sns.barplot(unv_df_2012_new['institution'], unv_df_2012_new['score'])

