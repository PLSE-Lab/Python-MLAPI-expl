#!/usr/bin/env python
# coding: utf-8

# # Imports 
# I'm going to use typical data science stack: numpy, pandas, sklearn, matplotlib.

# In[ ]:


# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns


# 
# # Read in Data
# 
# First, we can list all the available data files. There are a total of 9 files: 1 main file for training (with target) 1 main file for testing (without the target), 1 example submission file, and 6 other files containing additional information about each loan.
# 

# In[ ]:


# List files available
print(os.listdir("../input/"))


# In[ ]:


# Training data
app_train = pd.read_csv('../input/application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()


# As we can see our training dataset have total 122 features combinations of Ids, categorical variable and measures. It also includes TARGET variable which we want to predict.

# In[ ]:


# Testing data features
app_test = pd.read_csv('../input/application_test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head()


# The test set is considerably smaller and lacks a TARGET column.

# 
# # Exploratory Data Analysis
# 
# Exploratory Data Analysis (EDA) is an open-ended process where we calculate statistics and make figures to find trends, anomalies, patterns, or relationships within the data. The goal of EDA is to learn what our data can tell us. It generally starts out with a high level overview, then narrows in to specific areas as we find intriguing areas of the data. The findings may be interesting in their own right, or they can be used to inform our modeling choices, such as by helping us decide which features to use.
# 

# 
# # Examine the Distribution of the Target Column
# 
# The target is what we are asked to predict: either a 0 for the loan was repaid on time, or a 1 indicating the client had payment difficulties. We can first examine the number of loans falling into each category.
# 

# In[ ]:


app_train['TARGET'].value_counts()


# In[ ]:


app_train['TARGET'].plot.hist()


# From above plot, we see this is an [imbalanced class problem](http://chioka.in/class-imbalance-problem/). There are far more loans that were repaid on time than loans that were not repaid. Once we get into more sophisticated machine learning models, we can weight the classes by their representation in the data to reflect this imbalance.

# 
# # Examine Missing Values
# 
# Next we can look at the number and percentage of missing values in each column.
# 

# In[ ]:


app_train.info(verbose=True, null_counts=True)


# As we can see this gives us no clear picture about missing values that which colomns have how many missing  and what percentage of tatal values in that columns. Let's write a customised missing value table function

# In[ ]:


# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " rows.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[ ]:


# Missing values statistics
missing_values = missing_values_table(app_train)
missing_values.head(41)


# There is a general approcah where we drop coloumns which have more than 50% missing values. So above table you know which columns you can drop if you decided to drop it.
# When it comes time to build our machine learning models, we will have to fill in these missing values (known as imputation). In later work, we will use models such as XGBoost that can [handle missing values with no need for imputation](https://stats.stackexchange.com/questions/235489/xgboost-can-handle-missing-data-in-the-forecasting-phase). Another option would be to drop columns with a high percentage of missing values, although it is impossible to know ahead of time if these columns will be helpful to our model. Therefore, we will keep all of the columns for now.

# In[ ]:


sns.heatmap(app_train.isnull(), cbar=False)


# 
# # Column Types
# 
# Let's look at the number of columns of each data type. int64 and float64 are numeric variables (which can be either discrete or continuous). object columns contain strings and are categorical features. .
# 

# In[ ]:


app_train.dtypes.value_counts()


# Let's now look at the number of unique entries in each of the object (categorical) columns.

# In[ ]:


# Number of unique classes that is level of categorical variable in each object column
app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# Most of the categorical variables have a relatively small number of unique entries. We will need to find a way to deal with these categorical variables because most machine learning models deals with only numerical input.
