#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **At First we're considering 'bank.csv' file as Framing the Data with Pandas And we will solve some common problems that most of the people are facing to solve :
# 1)what is the most common occupation in the 'Job' column?
# 2)what is the (average) mean 'Balanc' for people with martial status as married and also education as tertiary?
# 3)what is the mean "age" for diff. "Educatin" categories?
# 4)Drop the following columns from the dataFrame:Job, day, month, previous, pdays,poutcome?
# 5)Recode categorical variables with 2 levels as 0& 1.?
# 6)replace the categorical variables with more than 2 levels with their rep.dummy variables?
# **

# In[ ]:


import pandas as pd
import missingno as msno


# In[ ]:


bank_data = pd.read_csv("../input/bank.csv",sep=";")
bank_data.head()


# In[ ]:


bank_data.tail()


# In[ ]:


msno.matrix(bank_data)


# there are 18 columns and 4521 rows in each columns

# In[ ]:


#lets find out numerical features in the Bank_data
numeric_features = bank_data.select_dtypes(include=[np.number])

numeric_features.columns


# In[ ]:


categorical_features = bank_data.select_dtypes(include=[np.object])

categorical_features.columns


# **1.a. What's the most common occupation in the 'Job' Column?**

# In[ ]:





# In[ ]:


x = bank_data.job.value_counts()
print(x.idxmax()) #solution 1


# In[ ]:


print(x[x == x.max()]) 


# In[ ]:


print(x[x == x.max()].index[0]) #solution 3


# **2.b. what's the mean (avg.) 'balance' for people with marital status as married and also education as tertiary?**
# 

# In[ ]:


bank_data.marital.value_counts()


# In[ ]:


bank_data.education.value_counts()


# In[ ]:


# above checks are to see if the required levels are represented in any other format
bank_data.loc[(bank_data.marital == 'married') & (bank_data.education == 'tertiary'),'balance'].mean() #solution1


# In[ ]:


#bank balance for marital status as 'single' & the customers who already finished their 'tertiary ' education
bank_data.loc[(bank_data.marital == 'single') & (bank_data.education == 'tertiary'), 'balance'].mean() #solution 2


# In[ ]:


#Bank balance for those whose marital status as ' divorced' & those people who already finished their 'tertiary' education
bank_data.loc[(bank_data.marital == "divorced") & (bank_data.education == 'tertiary'), 'balance'].mean() #solution 3


# In[ ]:


#Bank balane for marital status as 'married' & the customer who have done only 'primary' education

bank_data.loc[(bank_data.marital == 'married') & (bank_data.education == 'primary'),'balance'].mean() #solution


# In[ ]:


#Bank balance for those are divorsed & the customer done only 'primary' education
bank_data.loc[(bank_data.marital == 'divorced') & (bank_data.education == 'primary'),'balance'].mean() #solution


# **2.c. what is the mean "age" for diff. 'education' categories?**

# In[ ]:


bank_data.groupby('education').agg({'age':'mean'})


# **2.d. Drop the following columns from the dataframe: job, day, momth, pdays, previous, poutcome**

# In[ ]:


bank_data.drop(['job', 'day', 'month', 'pdays', 'previous', 'poutcome'],axis=1, inplace=True)
bank_data.head()


# **Recode the categorical variables with 2 levels as 0 & 1**

# In[ ]:


cat_cols = bank_data.columns[bank_data.dtypes == 'object']
num_cols = bank_data.columns[bank_data.dtypes == 'object']

two_level_cols = []
more_level_cols = []

for col in cat_cols:
    if len(pd.unique(bank_data[col])) ==2:
        two_level_cols.extend([col])
    else:
        more_level_cols.extend([col])
print(two_level_cols)
print(more_level_cols)


# In[ ]:


for col in two_level_cols:
    uq_values = pd.unique(bank_data[col])
    print(uq_values)
    bank_data.loc[bank_data[col]==uq_values[0], col] = 0
    bank_data.loc[bank_data[col]==uq_values[1], col] = 1
    print(pd.unique(bank_data[col]))


# **2.f. Replace categorical variables with more than 2 levels with their repective dummy variables**

# In[ ]:


bank_data = pd.get_dummies(bank_data, columns=more_level_cols)
bank_data.head()

