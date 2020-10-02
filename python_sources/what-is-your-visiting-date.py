#!/usr/bin/env python
# coding: utf-8

# Please upVote, if you like the work.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sbn
pd.set_option('display.max_columns', 5000)
pd.set_option('display.max_rows',5000)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


family_df = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')
sample_sub = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv')


# In[ ]:


sample_sub.head()


# In[ ]:


family_df.head()


# In[ ]:


family_df.n_people.value_counts().plot(kind = 'bar',figsize = (15,7))


# In[ ]:


family_df.isnull().sum()


# In[ ]:


family_df[family_df['choice_0']> 100]


# Every one wants gifts just before X-mas day

# In[ ]:


sbn.pairplot(family_df)


# In[ ]:


sbn.distplot(family_df.choice_0)


# In[ ]:


plt.figure(figsize = (16,5))
sbn.distplot(family_df.choice_1)


# In[ ]:


plt.figure(figsize=(16, 6))
ax = sbn.distplot(family_df.choice_1)


# In[ ]:


plt.figure(figsize = (16,6))
sbn.boxplot(x = family_df.n_people,y = family_df.choice_0)


# In[ ]:


plt.figure(figsize = (16,6))
sbn.boxplot(x = family_df.n_people,y = family_df.choice_9)


# In[ ]:


plt.figure(figsize = (16,6))
sbn.boxplot(x = family_df.n_people,y = family_df.choice_2)


# In[ ]:


family_df[family_df.choice_0==1]['n_people'].sum()


# In[ ]:


family_df.groupby(['choice_0'])['n_people'].agg(sum).plot(kind = 'bar',figsize = (16,6))


# Check any family's choice is repeatative

# In[ ]:



for index, row in family_df.iterrows():
    #print(row)
    hash_value = {}
    for (columnName, columnData) in row.iteritems():
        if (columnName != 'family_id') & (columnName != 'n_people'):
            hash_value[columnData] = 1+ hash_value.get(columnData,0)
            if hash_value[columnData] > 1:
                row[columnName] = -1
                #print(columnName)

            
 


# No repetative choices from familys'

# In[ ]:


family_df.values[(family_df == -1).values]


# In[ ]:


family_df.groupby(['choice_1'])['n_people'].agg(sum).plot(kind = 'bar',figsize = (16,6))


# In[ ]:


family_df.groupby(['choice_2'])['n_people'].agg(sum).plot(kind = 'bar',figsize = (16,6))


# In[ ]:


days_people = {}
for index, row in family_df.iterrows():
    for (columnName, columnData) in row.iteritems():
        if (columnName != 'family_id') & (columnName != 'n_people'):
            days_people["days_"+str(columnData )] = row['n_people'] + days_people.get("days_"+str(columnData),0)


               


# In[ ]:


days_people_df = pd.DataFrame.from_dict(list(days_people.items()))
days_people_df.columns = ['days_before_xmas','n_people_interested']


# All Peoples Interests Vs Visiting Day
# Most of the people interested in 1 Day before X-Mas

# In[ ]:


days_people_df.sort_values(by=['n_people_interested'], ascending=False).plot(x = 'days_before_xmas',y ='n_people_interested',  kind = 'bar',figsize = (16,6),)


# In[ ]:


(family_df.n_people.sum(),family_df.shape)


# Number of family interests Vs Days
# * More than 50% of families expecting day 1 should visit.

# In[ ]:


days_family = {}
for index, row in family_df.iterrows():
    for (columnName, columnData) in row.iteritems():
        if (columnName != 'family_id') & (columnName != 'n_people'):
            days_family["days_"+str(columnData )] = 1 + days_family.get("days_"+str(columnData),0)
            
days_family_df = pd.DataFrame.from_dict(list(days_family.items()))
days_family_df.columns = ['days_before_xmas','n_family_interested']


# In[ ]:


days_family_df.sort_values(by=['n_family_interested'], ascending=False).plot(x = 'days_before_xmas',y ='n_family_interested',  kind = 'bar',figsize = (16,6),)


# In[ ]:


family_df.n_people.sum()


# In[ ]:


def check_lowest_cost_day(row,n_alloc):
    flag = 0
    min_people = 100000
    day = 101
    for columnName,columnData in row.iteritems():
        if (columnName != 'family_id') and (columnName != 'n_people'):
            if flag == 0:
                flag = 1
                min_ = n_alloc.get(columnData,0) + row['n_people'] 
            else:
                min_ = n_alloc.get(columnData,0) + row['n_people'] 
            if (min_people > min_) and (min_ < 300):
                min_people = min_
                day = columnData
    
    if min_people == 100000:
        #print("outer")
        return min(n_allocated, key=n_allocated.get)
    return day
                        
            


# In[ ]:


results_dict = {}
n_allocated = {i:0 for i in range(1,101)}

for index, row in family_df.iterrows():
    day = check_lowest_cost_day(row,n_allocated)
    results_dict[row['family_id']] = day
    n_allocated[day] = n_allocated.get(day,0) + row['n_people']
    


# In[ ]:


results = pd.DataFrame.from_dict(list(results_dict.items()))
results.columns = ['family_id','assigned_day']


# In[ ]:


results_famil_df = results.merge(family_df,on='family_id',how = 'inner')


# In[ ]:


results_famil_df.groupby(['assigned_day'])['n_people'].agg(sum)


# In[ ]:


results.assigned_day.value_counts().plot(kind = 'bar', figsize = (16,6))


# In[ ]:


results.to_csv('submission.csv', index=False)


# In[ ]:




