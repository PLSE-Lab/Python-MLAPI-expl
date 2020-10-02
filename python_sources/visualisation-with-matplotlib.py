#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# 

# reading data files

# In[ ]:


movies_data=pd.read_csv("../input/tmdb_5000_movies.csv")
credits_data=pd.read_csv("../input/tmdb_5000_credits.csv")
movies_data=movies_data[movies_data["vote_count"]>1000]


# In[ ]:


movies_data.info()


# In[ ]:


credits_data.info()


# 

# visualise the data by using heatmap of seaborn

# In[ ]:


f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(movies_data.corr(),annot=True, linewidths=.5, fmt= '.2f',ax=ax,square=True)
plt.show()


# 

# printing the columns of data

# In[ ]:


movies_data.columns


# 

# plotting the correlation of budget and revenue by using scatter plot

# In[ ]:


movies_data.plot(kind='scatter', x='budget', y='revenue',alpha =0.6,color = 'red',figsize=(10,10))
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.title('Budget - Revenue Correlation') 
plt.show()


# 

# filtering the data by budget and revenue of those higher than 1 billion $

# In[ ]:


movies_data[(movies_data["budget"]>100000000) & (movies_data["revenue"]>1000000000)]


# 

# * finding the threshold of data
# * evaluating the first 20 movies by threshold of revenue

# In[ ]:


threshold = sum(movies_data.revenue)/len(movies_data.revenue)
movies_data["revenues"] = ["high" if i > threshold else "low" for i in movies_data.revenue]
movies_data.loc[:20,["original_title","revenue","revenues"]]


# evaluating of original languages of movies

# In[ ]:


print(movies_data['original_language'].value_counts())


# 

# sorting movies by vote average

# In[ ]:


movies_data.sort_values("vote_average",axis=0,ascending=False,inplace=True)


# 

# plotting revenue with vote average by using boxplot

# In[ ]:


movies_data.boxplot(column="revenue",by="vote_average",figsize=(30,5))
plt.show()


# 

# melting first five rows of data by one column

# In[ ]:


new_data=movies_data.head()
pd.melt(frame=new_data, id_vars='original_title', value_vars=['vote_average'])


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


date_list=movies_data["release_date"]
datetime_object=pd.to_datetime(date_list)
movies_data["date"]=datetime_object
movies_data=movies_data.set_index("date")
new_data=movies_data.head()
new_data


# In[ ]:


movies_data.resample("A").mean()


# In[ ]:


movies_data.vote_average.plot(color = 'r',label = 'vote_average',linewidth=1, alpha = 0.6,grid = True,linestyle = '-',figsize=(30,5))


# In[ ]:




