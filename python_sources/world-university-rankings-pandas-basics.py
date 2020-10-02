#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv ('../input/cwurData.csv')


# In[ ]:


data.info()


# In[ ]:


data.head(15)


# In[ ]:


data.tail()


# In[ ]:


data.describe()


# In[ ]:


data.corr ()


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


data["national_tops"] = ["national_top5" if i <=5  else "NaN" for i in data.national_rank] 
# adding a new column "national_tops" to see the 5 top universities
datatop5 = data.loc[:,["world_rank","institution","country","national_tops"]] #gives the info of  country names, institution and their national top 5
datatop5


    


# In[ ]:


data_new = data.head(10)    # I only take 10 rows into new data
data_new


# In[ ]:


# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
#in this example 
melted = pd.melt(frame=data_new,id_vars = 'institution', value_vars= ['country','national_rank'])
melted


# In[ ]:


# Index is institution
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'institution', columns = 'variable',values='value')


# In[ ]:


data.set_index("national_tops", inplace=True)
data.loc[['national_top5'], ['world_rank','institution','country']]  # in order to see the only top5 institutions  for the countries 


# In[ ]:


data = pd.read_csv ('../input/cwurData.csv')


# In[ ]:


#Boxplot to see if there are outliers in features publication, influence and citation. 
#However all the data shows the rank, it is nearly impossible to find the outliers. 
#If the data has been prepared according to the numbers, it would be possible to find some outliers. 
# There are no outliers
data.boxplot(column='publications')
plt.show ()


# In[ ]:


data.boxplot(column='citations')
plt.show ()


# In[ ]:


data.boxplot(column='influence')
plt.show ()


# In[ ]:


# lets look frequency of data years 
print(data.year.value_counts(dropna =False))  # if there are nan values that also be counted


# In[ ]:


# Lets look frequency of countries 
print(data.country.value_counts(dropna =False))  # if there are nan values that also be counted


# In[ ]:


data.dtypes


# In[ ]:


# lets convert datatypes to categorical, int or float.
data.score = data.score.astype('int64')
data.year = data.year.astype('category')
data.patents = data.patents.astype('float')


# In[ ]:


data.dtypes


# In[ ]:


# Top 50 institutions 
data["patent_level"] = ["high" if i <= 100  else "low" for i in data.patents]
data1 = data.loc[:49,["patent_level","institution"]] 


# In[ ]:


data["world_rank_level"] = ["high" if i <=100  else "low" for i in data.world_rank]
data2 = data.loc[:49,["world_rank_level","institution"]] 


# In[ ]:


data["publication_level"] = ["high" if i <= 100 else "low" for i in data.publications]
data3 = data.loc[:49,["publication_level","institution"]]


# In[ ]:


data["citation_level"] = ["high" if i <= 100  else "low" for i in data.citations]
data4 = data.loc[:49,["citation_level","institution"]] 


# In[ ]:


# to obtain a data set shows first 50 institutions according to the indicated levels.
# For example Princeton University, University of Chicago and 3 more have lower patent levels despite of being in top 50 institutions.
first50 = pd.concat([data1.institution,data1.patent_level, data2.world_rank_level,data3.publication_level,data4.citation_level],axis =1, ignore_index =True)
first50.columns = ['institution','patent_level','world_rank_level','pubication_level','citation_level']                                                                                                                                                 # axis = 0 : adds dataframes in row
first50  
# 


# In[ ]:




