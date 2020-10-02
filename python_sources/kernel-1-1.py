#!/usr/bin/env python
# coding: utf-8

# In[21]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## 1. Reading the data

# In[7]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.head()


# In[8]:


train_data.shape, test_data.shape


# ## 2. Data Exploration and Preprocessing

# ### 2.1 Univariate Analysis of target variable
# 
# 1. SalePrice see the distribution of the sale price - check whether it is normal or not (homoscedescity) 

# In[12]:


sns.distplot(train_data['SalePrice'])
plt.xticks(rotation=90)
plt.show()


# In[29]:


## plotting the normal probability plot

sales = train_data['SalePrice'].sort_values()
std_z_values = sorted(np.random.randn(len(sales)))

plt.scatter(std_z_values,sales)


# **Observations:**
# 
# The distribution plot is positively skewed and deviate from normal distribution
# 
# *for plotting the normal probability plot -* 
# https://github.com/kavyajeetbora/thinkstats/blob/master/5.%20Modelling%20distribution/normal_distribution.ipynb
# 

# In[14]:


print('Skewness: {:.2f} '.format(train_data['SalePrice'].skew()))
print('Kurtosis: {:.2f} '.format(train_data['SalePrice'].kurt()))


# ### 2.2. Multivariate Analysis
# 
# Here we try to understand the correlation of each independent variable with the dependent i.e. the SalePrice variable. 
# 
# 1. Using correlation matrix to find for the highly correlated independent variables - analysing and removing one of the variables 
# 2. Checking the correlation with the dependent variables -> here high correlation means its good for the model and better to keep that particular variable

# In[33]:


## scatter plot between the numerical variables with sales price
num_columns = []
for column in train_data.columns:
    if train_data[column].dtype != object:
        num_columns.append(column)
        
sns.scatterplot(train_data['TotalBsmtSF'], train_data['SalePrice'])


# Observations -
# 1. The correlation between the TotalBsmtSF and SalePrice looks strong 
# 2. But the variance is not uniform along the values - not homoscedescity

# In[35]:


#3 check the correlation of categorical variable with SalePrice

sns.boxplot(train_data['OverallQual'], train_data['SalePrice'])


# In[39]:


train_data['YearBuilt'].dtype


# Consider yearbuilt as categorical variable
# https://www.quora.com/Is-year-a-quantitative-or-categorical-variable

# In[38]:


plt.figure(figsize=(15,10))
sns.boxplot(train_data['YearBuilt'], train_data['SalePrice'])
plt.xticks(rotation=90)
plt.show()


# ### 2.3 Correlation between numerical variables

# In[ ]:





# ### 2.3 Basic Cleaning
# 1. Look for missing data along the columns - all remove the variables if required
# 2. Look for missing data along the rows
# 3. Look for outliers in the dataset after removing the nan values and remove the outliars as it can significantly affect out model - check the outliers using multivariate scatter plot

# In[ ]:


## Determing the number of null values per columns
def return_nan_columns(df,threshold=100):
    columns_to_remove = []
    no_of_null_values = {}
    for col in df.columns:
        series = df[col]
        null_values = len(series)-np.sum(series.notna())
        if null_values > 0:
            no_of_null_values.setdefault(col,null_values)
        if null_values > threshold:
            columns_to_remove.append(col)
    return columns_to_remove, no_of_null_values

columns_to_remove, no_of_null_values = return_nan_columns(train_data)
plt.bar(no_of_null_values.keys(),no_of_null_values.values())
plt.xticks(list(no_of_null_values.keys()),rotation=90)
plt.show()


# In[ ]:


# remove the columns with nan values more than some threshold = 500
train_data = train_data.drop(columns=columns_to_remove)
test_data = test_data.drop(columns=columns_to_remove)
train_data.head()


# In[ ]:


train_data.shape, test_data.shape


# 1. ## 2.2 Dealing with missing/NaN values in each rows

# In[ ]:


def show_nan_values(df):
    histogram = {}
    for col in df.columns:

        data = df[col]
        total_none = np.sum(data.isna())
        if total_none > 0:
            histogram.setdefault(col,total_none)

    plt.bar(histogram.keys(),histogram.values())
    plt.xticks(rotation=90)
    plt.show()
    
show_nan_values(train_data)


# In[ ]:


## filling the nan values by model
train_data.mode().iloc[0].head()


# In[ ]:


## filling the missing values for each columns with the mode value of the respective columns
def fill_na_by_mode(df):
    return df.fillna(df.mode().iloc[0])

train_data = fill_na_by_mode(train_data)
test_data = fill_na_by_mode(test_data)

print('Shape of the datasets:',train_data.shape, test_data.shape)
print('Number of nan values in train set: {:.0f} and test set: {:.0f}'.format(np.sum(np.sum(train_data.isna())),np.sum(np.sum(test_data.isna()))))


# In[ ]:


parameters = train_data.iloc[:,0:-1]
price = train_data.iloc[:,-1]

parameters.shape, test_data.shape


# ### 2.2 Divide the columns into numerical and categorical

# In[ ]:


## divide the dataset into numerical and categorical
def classify_columns(data, categorical=True):
    columns = []
    for col in data.columns:
        if categorical:  
            if data[col].dtype == object or col=='Id':
                columns.append(col)
                
        else:
            if data[col].dtype != object:
                columns.append(col)
    return columns

numerical_data = parameters[classify_columns(parameters, categorical=False)]
categorical_data = parameters[classify_columns(parameters)]

numerical_test_data = test_data[classify_columns(test_data, categorical=False)]
categorical_test_data = test_data[classify_columns(test_data)]

total = len(numerical_data.columns) + len(categorical_data.columns)
total_test = len(numerical_test_data.columns)+len(categorical_test_data.columns)
print('total train numerical variables: {:.0f} and categorial: {:.0f} out of total: {:.0f}'.format(len(numerical_data.columns),len(categorical_data.columns),total))
print('total test numerical variables: {:.0f} and categorial: {:.0f} out of total: {:.0f}'.format(len(numerical_test_data.columns),len(categorical_test_data.columns),total_test))

print(numerical_data.shape, categorical_data.shape, numerical_test_data.shape, categorical_test_data.shape)


# In[ ]:


categorical_data.head()


# ### 2.3 Normalization of the numerical data

# In[ ]:


def normalize_data_frame(df):
    norm_dict = {}
    for col in df.columns:
        if col != 'Id':
            mean = np.mean(df[col])
            std = np.std(df[col])
            norm_dict[col] = (mean,std)
            df[col] = (df[col]-mean)/std
    return df, norm_dict

X_norm, norm_dict = normalize_data_frame(numerical_data)
X_test_norm, norm_test_dict = normalize_data_frame(numerical_test_data)

print(X_norm.shape, X_test_norm.shape)

X_norm.head()


# ### 2.4 One Hot Encoding of the categorical data

# In[ ]:


## One Hot Encoding of the categorical dataset
X_cat_OH = pd.get_dummies(categorical_data)
X_cat_test_OH = pd.get_dummies(categorical_test_data)

print(X_cat_OH.shape, X_cat_test_OH.shape)

X_cat_OH.head()


# **How to use pd.get_dummies() with the test set** http://fastml.com/how-to-use-pd-dot-get-dummies-with-the-test-set/
# 
# = add the missing columns, filled with zeros

# In[ ]:


all_cols = list(X_cat_test_OH.columns) + list(X_cat_OH.columns)
all_unique_cols = list(set(all_cols))
print(len(all_unique_cols))


# In[ ]:


def add_missing_cols(df):
    missing_cols = []
    for cols in all_unique_cols:
        if cols not in df:
            missing_cols.append(cols)

    if len(missing_cols) > 0:
        missing_data = pd.DataFrame(np.zeros(shape=(len(df), len(missing_cols)),dtype=np.uint8),columns=missing_cols)
        missing_data['Id'] = df['Id'].values
        df = df.merge(missing_data,on='Id')
    return df

# missing_cols = return_missing_cols(X_cat_OH, X_cat_test_OH)
# missing_data = pd.DataFrame(np.zeros(shape=(len(X_cat_test_OH), len(missing_cols)),dtype=np.uint8),columns=missing_cols)
# missing_data['Id'] = X_cat_test_OH['Id'].values
# X_cat_test_OH.merge(missing_data,on='Id')

X_cat_OH = add_missing_cols(X_cat_OH)
X_cat_test_OH = add_missing_cols(X_cat_test_OH)
X_cat_OH = X_cat_OH.reindex(sorted(X_cat_OH.columns), axis=1)
X_cat_test_OH = X_cat_test_OH.reindex(sorted(X_cat_OH.columns), axis=1)


X_cat_OH.shape, X_cat_test_OH.shape


# ![](http://)Understanding difference between concatenate and Join function: https://qlikviewcookbook.com/2009/11/understanding-join-and-concatenate/
# 
# 1.  Concatenate never merges any rows - hence we it return the len(rows1)+len(rows2)
# 2.  Join merges the rows based on the key or index values - hence it will merge the rows and return same number of rows
# 
# Merging dataframes using pandas
# 
# https://www.shanelynn.ie/merge-join-dataframes-python-pandas-index-1/

# In[ ]:


#3 Now concatenate the datasets
joined_dataset = X_norm.merge(X_cat_OH,on='Id')
joined_test_dataset = X_test_norm.merge(X_cat_test_OH,on='Id')
joined_dataset.head()


# In[ ]:


X_train = joined_dataset.iloc[:,1:].values
X_test = joined_test_dataset.iloc[:,1:].values

X_train.shape, X_test.shape


# In[ ]:


# Normalizing the price 

sc = StandardScaler()
Y_norm = sc.fit_transform(price.values.reshape(-1,1))
plt.hist(Y_norm)
plt.xlabel('Normalized Price')
plt.show()


# ## 3. Modelling using machine learning models

# In[ ]:


get_ipython().run_cell_magic('time', '', 'regressor = RandomForestRegressor(n_estimators = 100)\nregressor.fit(X_train,Y_norm)')


# ## 4. Prediction

# In[ ]:


prediction = {'Id':joined_test_dataset['Id'] ,'SalePrice':sc.inverse_transform(regressor.predict(X_test).reshape(-1,1)).ravel().tolist()}
submission = pd.DataFrame(prediction)

submission.to_csv('submission.csv',index=False)

