#!/usr/bin/env python
# coding: utf-8

# # RCIC - Basic EDA v1

# Basic EDA of the train and test csv files for the Recursion Cellular Image Classification (RCIC) challenge on kaggle.

# ## Imports

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Unzip data and fix chmod

# In[ ]:


# list files in directory
get_ipython().system('ls -lh "../input"')


# In[ ]:


# unzip train.csv, is not needed in the kaggle kernel
#!unzip train.csv.zip


# In[ ]:


# set chmod to read train.csv, is not needed in the kaggle kernel
#!chmod +r train.csv


# ## Load data

# In[ ]:


# read csv data to pandas data frame
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# ## EDA for train and test dataset

# In[ ]:


# check for missing values
assert ~df_train.isnull().values.any()
assert ~df_test.isnull().values.any()

# check for NaN values
assert ~df_train.isna().values.any()
assert ~df_test.isna().values.any()


# In[ ]:


# check column name, non-null values and dtypes
print(df_train.info(),'\n')
print(df_test.info())


# In[ ]:


# have a look at the first rows
df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


# cast every column to object and get the unique elements
print(df_train.astype('object').describe(include='all').loc['unique', :],'\n')
print(df_test.astype('object').describe(include='all').loc['unique', :])


# The experiment count in the train and test set is different. We will look into this in more detail.

# In[ ]:


# get unique values of every column
col_values = [col for col in df_train]
unique_col_values_train = [df_train[col].unique() for col in df_train]
unique_col_values_test = [df_test[col].unique() for col in df_test]


# In[ ]:


# check if there are difference in the columns and if print them
for i, (c, a, b) in enumerate(zip(col_values, unique_col_values_train, unique_col_values_test)):
    
    if i == 0: continue # skip id_code
        
    a = set(a)
    b = set(b)
    
    print('\n'+c+':', a == b)
    
    # if the column elements are not equal, check if they are disjoint
    if not(a == b):
        print('disjoint:', a.isdisjoint(b))


# Based on this result, we can derive that the train and test dataset are based on different experiments.

# ## Visual EDA train data

# In[ ]:


sns.catplot(x='experiment', kind='count', data=df_train, height=3, aspect=10);


# The dataset count for all the experiments is comparable.

# In[ ]:


sns.catplot(x='plate', hue='experiment', kind='count', data=df_train, height=4, aspect=5);


# In every experiment we have 4 plates and the count is evenly distributed.

# In[ ]:


sns.catplot(y='well', kind='count', data=df_train, height=40, aspect=0.25);


# In[ ]:


sns.catplot(y='well', hue='experiment', kind='count', data=df_train, height=150, aspect=0.1);


# In general, the wells are evenly distributed. However, some wells are not present in every experiment, i.e., G3, G4, G11, G12, M08. (Note: If a experiment is not present in the first or the last position it can be tricky to see it in the visualisation due to the white background.)

# In[ ]:


sns.catplot(y='sirna', hue='experiment', kind='count', data=df_train, height=400, aspect=0.05);


# The siRNA data looks also very balanced. Nevertheless, some siRNAs were not used in every experiment.

# ## Visual EDA test dataset

# In[ ]:


sns.catplot(x='experiment', kind='count', data=df_test, height=3, aspect=10);


# The data set count for all the experiments is comparable.

# In[ ]:


sns.catplot(x='plate', hue='experiment', kind='count', data=df_test, height=4, aspect=5);


# In every experiment we have 4 plates. The count is evenly distributed.

# In[ ]:


sns.catplot(y='well', kind='count', data=df_test, height=40, aspect=0.25);


# In[ ]:


sns.catplot(y='well', hue='experiment', kind='count', data=df_test, height=150, aspect=0.1);


# In general, the wells are evenly distributed. However, some wells are not present in certain experiments.

# # Conclusion

# The train and test dataset contain different experiments. All in all, the train and test dataset looks very balanced.
