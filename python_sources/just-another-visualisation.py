#!/usr/bin/env python
# coding: utf-8

# Just another visualisation of the numerical variables present in the dataset..!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

data_path = "../input/"
train_file = data_path + "train_ver2.csv"
test_file = data_path + "test_ver2.csv"


# **Numerical variables:**
# 
# In this notebook, let us explore the 3 numerical variables present in the data.
# 
# 1. Age
# 2. Antiguedad - customer seniority
# 3. Renta
# 
# We can check the number of missing values, distribution of the data, distribution of the target variables based on the numerical variables in this notebook.
# 
# **Age:**

# In[ ]:


train = pd.read_csv(train_file, usecols=['age'])
train.head()


# In[ ]:


print(list(train.age.unique()))


# There are quite a few different formats for age (number, string with leading spaces, string). 
# 
# Also if we see, there is a **' NA'** value present in this field. So let us first take care of that by changing it to np.nan.

# In[ ]:


train['age'] = train['age'].replace(to_replace=[' NA'], value=np.nan)


# We can now convert the field to dtype 'float' and then get the counts

# In[ ]:


train['age'] = train['age'].astype('float64')

age_series = train.age.value_counts()
plt.figure(figsize=(12,4))
sns.barplot(age_series.index.astype('int'), age_series.values, alpha=0.8)
plt.ylabel('Number of Occurrences of the customer', fontsize=12)
plt.xlabel('Age', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# We could see that there is a very long tail at both the ends. So we can have min and max cap at some points respectively (I would use 20 and 86 from the graph). 

# In[ ]:


train.age.isnull().sum()


# In[ ]:


train.age.mean()


# We have 27734 missing values and the mean age is 40. We could probably do a mean imputation here. 
# 
# We could look at test set age distribution to confirm both train and test have same distribution.

# In[ ]:


test = pd.read_csv(test_file, usecols=['age'])
test['age'] = test['age'].replace(to_replace=[' NA'], value=np.nan)
test['age'] = test['age'].astype('float64')

age_series = test.age.value_counts()
plt.figure(figsize=(12,4))
sns.barplot(age_series.index.astype('int'), age_series.values, alpha=0.8)
plt.ylabel('Number of Occurrences of the customer', fontsize=12)
plt.xlabel('Age', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# Good to see that the distribution is similar between train and test.!
# 
# **ANTIGUEDAD:**
# 
# Customer seniority in months.

# In[ ]:


train = pd.read_csv(train_file, usecols=['antiguedad'])
train.head()


# In[ ]:


print(list(train.antiguedad.unique()))


# Here again we could see that there is a **'     NA'** value present in this field similar to age. Also we could see that there is a special value '-999999' present in the data. May be this special value also represent missing value?!
# 
# We shall first convert the NA value to np.nan value

# In[ ]:


train['antiguedad'] = train['antiguedad'].replace(to_replace=['     NA'], value=np.nan)
train.antiguedad.isnull().sum()


# So here again we have 27734 missing values.
# 
# We can convert the field to dtype 'float' and then check the count of special value -999999.

# In[ ]:


train['antiguedad'] = train['antiguedad'].astype('float64')
(train['antiguedad'] == -999999.0).sum()


# We have 38 special values. If we use a tree based model, we could probably leave it as such or if we use a linear model, we need to map it to mean or some value in the range of 0 to 256.
# 
# Now we can see the distribution plot of this variable.

# In[ ]:


col_series = train.antiguedad.value_counts()
plt.figure(figsize=(12,4))
sns.barplot(col_series.index.astype('int'), col_series.values, alpha=0.8)
plt.ylabel('Number of Occurrences of the customer', fontsize=12)
plt.xlabel('Customer Seniority', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# There are few peaks and troughs in the plot but there are no visible gaps or anything as such which is alarming (atleast to me.!)
# 
# So we can also see whether test follows a similar pattern and if it does then we are good.

# In[ ]:


test = pd.read_csv(test_file, usecols=['antiguedad'])
test['antiguedad'] = test['antiguedad'].replace(to_replace=[' NA'], value=np.nan)
test['antiguedad'] = test['antiguedad'].astype('float64')

col_series = test.antiguedad.value_counts()
plt.figure(figsize=(12,4))
sns.barplot(col_series.index.astype('int'), col_series.values, alpha=0.8)
plt.ylabel('Number of Occurrences of the customer', fontsize=12)
plt.xlabel('Customer Seniority', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# Peaks are comparatively bigger than the train set. Any implications?
# 
# **RENTA:**
# 
# Gross income of the household.

# In[ ]:


train = pd.read_csv(train_file, usecols=['renta'])
train.head()


# In[ ]:


unique_values = np.sort(train.renta.unique())
plt.scatter(range(len(unique_values)), unique_values)
plt.show()


# It seems the distribution of rent is highly skewed. There are few very high valued customers present in the data.
# 
# Let us get the mean and median value for this field.

# In[ ]:


train.renta.mean()


# In[ ]:


train.renta.median()


# Now let us see the number of missing values in this field.

# In[ ]:


train.renta.isnull().sum()


# There are quite a few number of missing values present in this field.! We can do some form of imputation for the same. One very good idea is given by Alan in this [script][1].
# 
# We can check the quantile distribution to see how the value changes in the last percentile.
# 
# 
#   [1]: https://www.kaggle.com/apryor6/santander-product-recommendation/detailed-cleaning-visualization-python

# In[ ]:


train.fillna(101850., inplace=True) #filling NA as median for now
quantile_series = train.renta.quantile(np.arange(0.99,1,0.001))
plt.figure(figsize=(12,4))
sns.barplot((quantile_series.index*100), quantile_series.values, alpha=0.8)
plt.ylabel('Rent value', fontsize=12)
plt.xlabel('Quantile value', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# As we can see there is a sudden increase in the rent value from 99.9% to 100%. So let us max cap the rent values at 99.9% and then get a box plot.

# In[ ]:


rent_max_cap = train.renta.quantile(0.999)
train['renta'][train['renta']>rent_max_cap] = 101850.0 # assigining median value 
sns.boxplot(train.renta.values)
plt.show()


# From the box plot, we can see that most of the rent values fall between 0 and 300,000.
# 
# Now we can see the distribution of rent in test data as well.

# In[ ]:


test = pd.read_csv(test_file, usecols=['renta'])
test['renta'] = test['renta'].replace(to_replace=['         NA'], value=np.nan).astype('float') # note that there is NA value in test
unique_values = np.sort(test.renta.unique())
plt.scatter(range(len(unique_values)), unique_values)
plt.show()


# *Please note that there is a new value '   NA' present in the test data set while it is not in train data.*
# 
# The distribution looks similar to train though.

# In[ ]:


test.renta.mean()


# In[ ]:


test.fillna(101850., inplace=True) #filling NA as median for now
quantile_series = test.renta.quantile(np.arange(0.99,1,0.001))
plt.figure(figsize=(12,4))
sns.barplot((quantile_series.index*100), quantile_series.values, alpha=0.8)
plt.ylabel('Rent value', fontsize=12)
plt.xlabel('Quantile value', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


test['renta'][test['renta']>rent_max_cap] = 101850.0 # assigining median value 
sns.boxplot(test.renta.values)
plt.show()


# So box and quantile plots are similar to that of the train dataset for rent.!
# 
# **Numerical variables Vs Target variables:**
# 
# Now let us see how the targets are distributed based on the numerical variables present in the data. Let us subset the first 100K rows for the same. 

# In[ ]:


train = pd.read_csv(data_path+"train_ver2.csv", nrows=100000)
target_cols = ['ind_cco_fin_ult1', 'ind_cder_fin_ult1',
                             'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                             'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
                             'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
                             'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                             'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',
                             'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
                             'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                             'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
                             'ind_viv_fin_ult1', 'ind_nomina_ult1',
                             'ind_nom_pens_ult1', 'ind_recibo_ult1']
train[target_cols] = (train[target_cols].fillna(0))
train["age"] = train['age'].map(str.strip).replace(['NA'], value=0).astype('float')
train["antiguedad"] = train["antiguedad"].map(str.strip)
train["antiguedad"] = train['antiguedad'].replace(['NA'], value=0).astype('float')
train["antiguedad"].ix[train["antiguedad"]>65] = 65 # there is one very high skewing the graph
train["renta"].ix[train["renta"]>1e6] = 1e6 # capping the higher values for better visualisation
train.fillna(-1, inplace=True)


# In[ ]:


fig = plt.figure(figsize=(16, 120))
numeric_cols = ['age', 'antiguedad', 'renta']
#for ind1, numeric_col in enumerate(numeric_cols):
plot_count = 0
for ind, target_col in enumerate(target_cols):
    for numeric_col in numeric_cols:
        plot_count += 1
        plt.subplot(22, 3, plot_count)
        sns.boxplot(x=target_col, y=numeric_col, data=train)
        plt.title(numeric_col+" Vs "+target_col)
plt.show()


# Seems all these numerical variables have some predictive power since they show some different behavior between 0's and 1's.
