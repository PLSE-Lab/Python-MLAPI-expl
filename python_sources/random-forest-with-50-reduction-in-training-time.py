#!/usr/bin/env python
# coding: utf-8

# Little attention is given to the actual types of data contained in pandas dataframes and how they can affect performance.
# But by carrying out some simple optimisations, training times can be dramatically reduced - in this example by 50%
# 
# In this short notebook i will:
# 1. Discuss the datatypes
# 2. Show how to determine and reduce the memory footprint of a dataframe
# 3. Run a normal and reduced memory footprint dataframe using a Random Forest algorithm with associated parameters and compare the execution times
# 
# This is not an optimal machine learning solution - but using similar techniques it should be possible to reduce your training times.
# 
# 
# References: https://www.dataquest.io/blog/pandas-big-data/
# 

# In[106]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[107]:


train = pd.read_csv('../input/train.csv', parse_dates=['project_submitted_datetime'])
test = pd.read_csv('../input/test.csv', parse_dates=['project_submitted_datetime'])
resources = pd.read_csv('../input/resources.csv')
#submission = pd.read_csv('sample_submission.csv')


# In[108]:


train.head(1)
resources.head()


# For demonstration purposes we select some data from the train and resources data.  Also add a new feature  

# In[109]:


mem_df1 = train[['id','teacher_id','teacher_number_of_previously_posted_projects','project_is_approved']]
resources['total_cost'] = resources['quantity']*resources['price']
mem_df2 = resources[['id','quantity','price','total_cost']]


# In[110]:


train_mem = pd.merge(mem_df1,mem_df2,how='left',on='id')
train_mem.info()


# We can see that we have strings, integers and floats.  Here we will concentrate on the numeric datatypes.  For the integer types what are the actual ranges?

# In[111]:


#some examples of datatypes unsigned and signed
data_types = ["uint8","int8","int16","uint16","uint64","int64"]
for it in data_types:
    print(np.iinfo(it))


# We can see in our dataframe we have three int64 datatypes, do we really need such a size? Lets check  

# In[112]:


train_mem.describe()


# We can see that teacher_number_of_previously_posted_projects ranges from 0 to 451, project_is_approved ranges for 0 to 1 and quantity ranges from 1 to 9999.  Lets select and convert the integer types using a simple function to see how much memory we could save: 

# In[113]:


def mem_usage(pandas_obj):
    usage_b = pandas_obj.memory_usage(deep=True).sum()
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

train_mem_int = train_mem.select_dtypes(include=['int'])
converted_int = train_mem_int.apply(pd.to_numeric,downcast='unsigned')

print("Size of integer types before {}".format(mem_usage(train_mem_int)))
print("Size of integer types after {}".format(mem_usage(converted_int)))

compare_ints = pd.concat([train_mem_int.dtypes,converted_int.dtypes],axis=1)
compare_ints.columns = ['No. of types before','No. of types after']
compare_ints.apply(pd.Series.value_counts)


# Lets repeat this for the float columns

# In[114]:


train_mem_float = train_mem.select_dtypes(include=['float'])
converted_float = train_mem_float.apply(pd.to_numeric,downcast='float')

print("Size of float types before: {}".format(mem_usage(train_mem_float)))
print("Size of float types after: {}".format(mem_usage(converted_float)))

compare_floats = pd.concat([train_mem_float.dtypes,converted_float.dtypes],axis=1)
compare_floats.columns = ['No. of types before','No. of types after']
compare_floats.apply(pd.Series.value_counts)


# Ok, lets try a simple gridsearch using random forest on the ORIGINAL datatype dataframe

# In[115]:


X = train_mem.drop(['id', 'project_is_approved','teacher_id'], axis=1)
y = train_mem['project_is_approved']
print(X.dtypes)
print(" ")
print(y.dtypes)


# In[116]:


num_folds = 5
seed = 7
scoring = 'accuracy'

start = time.time()
param_grid = {'max_depth': [5,8],
              'min_samples_split':[3,5]
             }

model = RandomForestClassifier(n_jobs=-1)
kfold =KFold(n_splits = num_folds,random_state = seed)
grid = GridSearchCV(estimator = model,param_grid = param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X,y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))
time_first = end - start


# Lets replace the int and float datatypes with the values calculated above

# In[117]:


columns_to_overwrite_float = ['total_cost','price']
train_mem.drop(labels=columns_to_overwrite_float, axis="columns", inplace=True)
train_mem[columns_to_overwrite_float] = converted_float[columns_to_overwrite_float]


# In[118]:


columns_to_overwrite_int = ['teacher_number_of_previously_posted_projects','project_is_approved','quantity']
train_mem.drop(labels=columns_to_overwrite_int, axis="columns", inplace=True)
train_mem[columns_to_overwrite_int] = converted_int[columns_to_overwrite_int]


# In[119]:


train_mem_after = train_mem.copy(deep=True)
train_mem_after.info()


# This new dataframe has a memory usage of 38 MB down from 66MB in the original

# In[120]:


X = train_mem_after.drop(['id', 'project_is_approved','teacher_id'], axis=1)
y = train_mem_after['project_is_approved']
print(X.dtypes)
print(" ")
print(y.dtypes)


# In[121]:


start = time.time()

param_grid = {'max_depth': [5,8],
              'min_samples_split':[3]
             }
model = RandomForestClassifier(n_jobs=-1)
kfold =KFold(n_splits = num_folds,random_state = seed)
grid = GridSearchCV(estimator = model,param_grid = param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X,y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))
time_second = end - start


# In[122]:


plt.figure(figsize=(10,5))
x_pos = [0,1]
x_label= ['before','after']
scores = [time_first,time_second]
plt.bar(x_pos,scores,align='center')
plt.xlabel('Memory Optimisation',fontsize=12)
plt.xticks(x_pos,x_label)
plt.ylabel('Time taken in secs',fontsize=12)
plt.show();


# We have reduced our training time by approximately 50% and still achieved the same accuracy score

# This simple example has demonstrated that by having greater understanding of the underlying data, it is possible to reduce the required processing times without any degradation in performance. 
# 
# If you found this notebook interesting please upvote it!

# In[ ]:




