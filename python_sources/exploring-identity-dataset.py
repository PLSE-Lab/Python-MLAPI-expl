#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import gc #garbage collection 
gc.enable()  #enabling garbage collection

#suppressing warnings.
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Data path
PATH = '/kaggle/input/ieee-fraud-detection/'


# In[ ]:


#Reading the train and test identity dataset.
train_identity = pd.read_csv(f'{PATH}train_identity.csv')
test_identity = pd.read_csv(f'{PATH}test_identity.csv')


# In[ ]:


#shape of the dataframe
print(f'Train Shape: {train_identity.shape}')
print(f'Test Shape: {test_identity.shape}')


# # Univariate Analysis.

# In[ ]:


#exploring first five observations.
train_identity.head()


# In[ ]:


#Exploring id_01.
train_identity.id_01.describe()


# id_01 - contains negative numbers with maximum value to be 0 and minimum number -100.
# 
# Contains no null values.

# In[ ]:


train_identity.id_01.value_counts(dropna=False)


# Contains 77 unique values.

# In[ ]:


#id_02.
train_identity.id_02.describe()


# In[ ]:


train_identity.id_02.value_counts(dropna=False)


# id_02 - Data is scattered. Maybe it is the credit limit or the max amount withdrawl limit. However the minimum value for id_02 is 1 which cannot be the withdrawl limit.

# In[ ]:


#distribution of the data.
sns.distplot(train_identity.id_02, bins=20);


# In[ ]:


#3. id_03.
train_identity.id_03.describe()


# id_03 - Value ranges from -13 to  10. Negative values are present in the column. Most of the values are 0 and NAN.

# In[ ]:


train_identity.id_03.value_counts(dropna=False)


# In[ ]:


#4.id_04.
train_identity.id_04.describe()


# **id_04** - Contains negative values, ranges from -28 to 0.

# In[ ]:


train_identity.id_04.value_counts(dropna=False)


# In[ ]:


#id_05.
train_identity.id_05.describe()


# In[ ]:


train_identity.id_05.value_counts(dropna=False)


# id_05 - Negative values present with most of the values to be 0 and NAN. Total 94 unique values.

# In[ ]:


#id_06.
train_identity.id_06.describe()


# In[ ]:


train_identity.id_06.value_counts(dropna=False)


# In[ ]:


#id_07.
train_identity.id_07.describe()


# In[ ]:


train_identity.id_07.value_counts(dropna=False)


# In[ ]:


#id_08
train_identity.id_08.describe()


# In[ ]:


train_identity.id_08.value_counts(dropna=False)


# In[ ]:


#id_09
train_identity.id_09.describe()


# In[ ]:


train_identity.id_09.value_counts(dropna=False)


# In[ ]:


#id_10
train_identity.id_10.describe()


# In[ ]:


train_identity.id_10.value_counts(dropna=False)


# In[ ]:


#id_11
train_identity.id_11.describe()


# In[ ]:


train_identity.id_11.value_counts(dropna=False)


# In[ ]:


#id_12
train_identity.id_12.describe()


# id_12 : Categorical Data. 2 Unique values, Found and Not Found.

# In[ ]:


#count plot.
sns.countplot(train_identity.id_12)
print(f'NAN count: {train_identity.id_12.isnull().sum()}')


# In[ ]:


#id_13
train_identity.id_13.describe()


# id_13 : Nominal Categorical variable containg of numbers representing some categories.

# In[ ]:


train_identity.id_13.value_counts(dropna=False)


# In[ ]:


#id_14
train_identity.id_14.describe()


# In[ ]:


train_identity.id_14.value_counts(dropna=False)


# In[ ]:


#id_15
train_identity.id_15.describe()


# In[ ]:


print(f'NAN count: {train_identity.id_15.isnull().sum()}')
sns.countplot(train_identity.id_15);


# In[ ]:


#id_16
train_identity.id_16.describe()


# In[ ]:


print(f'NAN count: {train_identity.id_16.isnull().sum()}')
sns.countplot(train_identity.id_16);


# In[ ]:


#id_17
train_identity.id_17.describe()


# In[ ]:


train_identity.id_17.value_counts(dropna=False)


# In[ ]:


#id_18.
train_identity.id_18.describe()


# In[ ]:


#id_19.
train_identity.id_19.describe()


# In[ ]:


#id_20.
train_identity.id_20.describe()


# In[ ]:


#id_21.
train_identity.id_21.describe()


# In[ ]:


#id_22.
train_identity.id_22.describe()


# In[ ]:


#id_23.
train_identity.id_23.describe()


# In[ ]:


print(f'NAN count: {train_identity.id_23.isnull().sum()}')
sns.countplot(train_identity.id_23)
plt.xticks(rotation=90);


# id_23 : Tells us about the IP proxy status of the transaction.

# In[ ]:


#id_24.
train_identity.id_24.describe()


# In[ ]:


#id_25.
train_identity.id_25.describe()


# In[ ]:


#id_26.
train_identity.id_26.describe()


# In[ ]:


#id_27.
train_identity.id_27.describe()


# In[ ]:


#id_28.
train_identity.id_28.describe()


# In[ ]:


#id_29.
train_identity.id_29.describe()


# In[ ]:


#id_30.
train_identity.id_30.describe()


# In[ ]:


print(f'NAN count: {train_identity.id_30.isnull().sum()}')
plt.figure(figsize=(13, 8))
sns.countplot(train_identity.id_30)
plt.xticks(rotation=90);


# id_30 - The OS version used for the transactions.

# In[ ]:


#id_31.
train_identity.id_31.describe()


# In[ ]:


print(f'NAN count: {train_identity.id_31.isnull().sum()}')
plt.figure(figsize=(19, 8))
sns.countplot(train_identity.id_31)
plt.xticks(rotation=90);


# id_31 : The Browser used for the transaction.

# In[ ]:


#id_32.
train_identity.id_32.describe()


# In[ ]:


#id_33.
train_identity.id_33.describe()


# In[ ]:


print(f'NAN count: {train_identity.id_33.isnull().sum()}')
plt.figure(figsize=(19, 8))
sns.countplot(train_identity.id_33.iloc[:30000])
plt.xticks(rotation=90);


# id_33 : Screen resolution of the device used. Can be a uselsess feature.

# In[ ]:


#id_34.
train_identity.id_34.describe()


# In[ ]:


print(f'NAN count: {train_identity.id_34.isnull().sum()}')
plt.figure(figsize=(9, 5))
sns.countplot(train_identity.id_34.iloc[:30000])
plt.xticks(rotation=90);


# In[ ]:


#id_35.
train_identity.id_35.describe()


# In[ ]:


print(f'NAN count: {train_identity.id_35.isnull().sum()}')
plt.figure(figsize=(9, 5))
sns.countplot(train_identity.id_35)
plt.xticks(rotation=90);


# In[ ]:


#id_36.
train_identity.id_36.describe()


# In[ ]:


print(f'NAN count: {train_identity.id_36.isnull().sum()}')
plt.figure(figsize=(9, 5))
sns.countplot(train_identity.id_36)
plt.xticks(rotation=90);


# In[ ]:


#id_37.
train_identity.id_37.describe()


# In[ ]:


print(f'NAN count: {train_identity.id_37.isnull().sum()}')
plt.figure(figsize=(9, 5))
sns.countplot(train_identity.id_37)
plt.xticks(rotation=90);


# In[ ]:


#id_38
train_identity.id_38.describe()


# In[ ]:


print(f'NAN count: {train_identity.id_38.isnull().sum()}')
plt.figure(figsize=(9, 5))
sns.countplot(train_identity.id_38)
plt.xticks(rotation=90);


# In[ ]:


#Device Info
train_identity.DeviceInfo.describe()


# In[ ]:


print(f'NAN count: {train_identity.DeviceInfo.isnull().sum()}')
plt.figure(figsize=(19, 5))
sns.countplot(train_identity.DeviceInfo[:500])
plt.xticks(rotation=90);


# DeviceInfo: Info about the device used for the transaction.

# In[ ]:


#Device Type
train_identity.DeviceType.describe()


# In[ ]:


print(f'NAN count: {train_identity.DeviceType.isnull().sum()}')
plt.figure(figsize=(9, 5))
sns.countplot(train_identity.DeviceType)
plt.xticks(rotation=90);


# # Bivariate analysis.

# In[ ]:


cols = ['id_0'+str(i) for i in range(1, 10)]
cols += ['id_10', 'id_11']

#selecting only numerical columns
train_num = train_identity[cols]
#correlation
corr = train_num.corr()

#heatmap
plt.figure(figsize=(15,9))
sns.heatmap(corr, annot=True, vmax=1., vmin=-1.)


# id_03 and id_09 are highly corelated with each other.

# In[ ]:


#distribution.
sns.distplot(train_identity.id_03.dropna())


# In[ ]:


sns.distplot(train_identity.id_09.dropna())


# The distribution is similar.

# In[ ]:


#All variables correlation.
cols = [col for col in train_identity.columns if train_identity[col].dtype != 'O']
cols.remove('TransactionID')

#selecting only numerical columns
train_num = train_identity[cols]
#correlation
corr = train_num.corr()

#heatmap
plt.figure(figsize=(21, 21))
sns.heatmap(corr, annot=True, vmax=1., vmin=-1., cmap='mako')


# In[ ]:


#checking the categrical variables containing in the train sets are also present in the test set.
def checkcat(df):
    for col in df.columns:
        length = len(set(test_identity[col].values) - set(train_identity[col].values))
        if length > 0:
            print(f'{col} in the test set has {length} values that are not present in the train set')

cat_cols = [col for col in train_identity.columns if train_identity[col].dtype == 'O']
#cat_cols += ['id_13', 'id_14', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22',
#           'id_24', 'id_25', 'id_26', 'id_32']
print(cat_cols)


# In[ ]:


checkcat(train_identity[cat_cols])


# In[ ]:




