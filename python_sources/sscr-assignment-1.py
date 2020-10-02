#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))


# In[ ]:


# Reading Dataset

Zoo=pd.read_csv("../input/zoo.csv")
Zoo.head()


# In[ ]:


# Task Number 1
# Aggregating Data
# 1) Count
# 2) Sum
# 3) Min, Max
# 4) Mean, Median
# 5) Grouping

Zoo.count()


# In[ ]:


Zoo[['animal']].count()   # Count For Animal Only


# In[ ]:


Zoo.animal.count()


# In[ ]:


Zoo.water_need.sum()


# In[ ]:


Zoo.water_need.max()


# In[ ]:


Zoo.water_need.min()


# In[ ]:


Zoo.groupby('animal').mean()    # Run aggregation function for all columns


# In[ ]:


Zoo.groupby('animal').mean().water_need


# In[ ]:


Zoo.groupby('animal').mean()[['water_need']]


# In[ ]:


#'water_need': lambda x: x.max() - x.min()
def max_min(x):
    return x.max() - x.min()


Zoo.groupby('animal').agg({'uniq_id':'count', 
                           'water_need':['mean', 'max', 'min', max_min]
                         })


# In[ ]:


Zoo[Zoo.animal == 'zebra'].groupby('animal').agg({'uniq_id':'count', 
                           'water_need':['mean', 'max', 'min', max_min]
                         })


# In[ ]:


# Transpose of a Dataset
# Create the index 
Rows = ['Row_'] * 22
Rows = list(Rows)
Numbers = list(range(1,23))
concat_func = lambda x,y: x + "" + str(y)
index_ = list(map(concat_func,Rows,Numbers))

# Apply Index to Zoo Data
Zoo.index = index_ 
print(Zoo) 
#index_ = map(lambda (x,y): zip(Rows,Numbers))
#print(index_)


# In[ ]:


Zoo_Tr = Zoo.transpose()
Zoo_Tr


# In[ ]:


data = {'ID':[1,2,3,4,5,6,7], 'Year':[2016,2016,2016,2016,2017,2017,2017], 'Jan_salary':[4500,4200,4700,4500,4200,4700,5000], 'Feb_salary':[3800,3600,4400,4100,4400,5000,4300], 'Mar_salary':[3700,3800,4200,4700,4600,5200,4900]}
df = pd.DataFrame(data)
df


# In[ ]:


# Wide to Long Dataset

melted_df = pd.melt(df,id_vars=['ID','Year'],
                        value_vars=['Jan_salary','Feb_salary','Mar_salary'],
                        var_name='month',value_name='salary')
melted_df


# In[ ]:


# Long To Wide
Casted_df = melted_df.pivot_table(index=['ID','Year'], columns='month', values='salary', aggfunc='first')
Casted_df


# In[ ]:


# Crosstab
pd.crosstab(index=[melted_df['Year']], columns=[melted_df['month']])


# In[ ]:


# head
Zoo.head(10)


# In[ ]:


# Tail
Zoo.tail(4)


# In[ ]:


# Select/Drop variables & Selecting Observations
df.loc[:, ['ID', 'Year','Jan_salary']]


# In[ ]:


df.loc[:3, ['ID', 'Year','Jan_salary']]


# In[ ]:


df.loc[2:5, ['ID', 'Year','Jan_salary']]


# In[ ]:


df.drop('Jan_salary', axis=1)


# In[ ]:


df.drop(df.index[2])


# In[ ]:


# Rndom sample with replacement

#Ind = random.choices(list(range(1,23)), k=12)     # Size = 12
Ind = random.choices(Zoo.index, k=12)     # Size = 12
Sample_1 = Zoo.loc[Ind,]
Sample_1


# In[ ]:


# Rndom sample with replacement
Ind = random.sample(list(Zoo.index), k=12)
Sample_2 = Zoo.loc[Ind,]
Sample_2

