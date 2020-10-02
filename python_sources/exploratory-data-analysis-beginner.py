#!/usr/bin/env python
# coding: utf-8

# # Objective

# To do Exploratory Data Analysis. Try to do following steps
# 1. Understanding the Dataset.
#     * Size of the dataset
#     * Datatype of each column
#     * 5 summary statistics
#     * Target variable analysis
# 2. Clean the data.
# 3. Relationship analysis.
#     * Histogram
#     * Pair Plot
#     * Joint Plot

# 

# # Importing Libraries

# We are importing Matplotlib and Seaborn Libraries for Data Visualisation.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import seaborn as sns # for data visualisation
import matplotlib.pyplot as plt # for data visualisation


# In[ ]:


df=pd.read_csv("../input/habermans-survival-data-set/haberman.csv")
df.head(5)


# From inspecting the dataframe. We know that no column name. so we need to add column name to dataframe.

# In[ ]:


df=pd.read_csv("../input/habermans-survival-data-set/haberman.csv",names=['age','op_year','axil_nodes','sur_status'])
df.head(5)


# # 1. Understanding the data

# In[ ]:


print(df.shape)
print("Dataset has {} rows and {} columns".format(df.shape[0],df.shape[1]))


# In[ ]:


df.info()


# * Index column is RangeIndex.
# * All other columns are Int64 Data type.
# * We know that sur_status is the target variable.
# * All other columns having discrete values.

# In[ ]:


df.describe()


# * There is no missing values in the above columns. Because the count of each column is matching with the row count of dataframe.
# * Patient age is from 30 to 83
# * Op_year have data from year 1958 to 1969.
# * More than 75% of patients have less than 5 nodes. eventough Maximum value is 52.

# In[ ]:


df['sur_status'].value_counts()


# * Dataset is unbalanced. 
# * 225 patients are survived and 81 patients are not survived.

# # Data Cleaning

# * From Previous exploration there is No Missing Values.
# * We will check whether the missing values are represented in some other value.

# In[ ]:


df.nunique()


# * Age has 49 Unique columns
# * Operating year has 12 Unique columns
# * Nodes has 31 Unique columns

# In[ ]:


df['age'].unique()


# In[ ]:


df['op_year'].unique()


# In[ ]:


df['axil_nodes'].unique()


# In[ ]:


df['sur_status'].unique()


# From Looking the unique values, Nan values are not coded in the different values.

# # Relationship Analysis

# **Univariant Analysis**

# Creating Histogram using Seaborn

# In[ ]:


plt.figure(figsize=[16,16])
plt.subplot(221)
sns.distplot(df['age'])
plt.subplot(222)
sns.distplot(df['op_year'])
plt.subplot(223)
sns.distplot(df['axil_nodes'])
plt.subplot(224)
sns.distplot(df['sur_status'])
plt.show()


# * Age column is Normal Distributed.
# * Axil_nodes is Right Skewed.

# **Box Plot**

# In[ ]:


sns.boxplot(x='sur_status',y='age', data=df)
plt.show()


# In[ ]:


sns.boxplot(x='sur_status',y='axil_nodes', data=df)
plt.show()


# Axil_Nodes column has few values that are exceptional.

# In[ ]:


sns.boxplot(x='sur_status',y='age', data=df)
plt.show()


# **Pair Plot**

# In[ ]:


sns.pairplot(df,hue="sur_status",height=3)


# **Joint Plot**

# In[ ]:


sns.jointplot(x='sur_status',y='age', data=df, kind="kde");


# In[ ]:


sns.jointplot(x='sur_status',y='op_year', data=df, kind="kde");


# In[ ]:


sns.jointplot(x='sur_status',y='axil_nodes', data=df, kind="kde");

