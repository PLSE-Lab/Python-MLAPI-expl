#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

# other libraries and functions
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# ## Read in Data

# In[ ]:


df = pd.read_csv("../input/application_train.csv")


# In[ ]:


train, test = train_test_split(df, test_size=0.2)


# In[ ]:


train.shape


# In[ ]:


test.shape


# ## Peak at Data

# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Check Missing Values

# First, define a function to build a nice table of missing value percentages.  This will make things easier given the large number of initial features.

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
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[ ]:


# Missing values statistics
missing_train = missing_values_table(train)
missing_train.head(10)


# # Data Types

# In[ ]:


# Number of each type of column
train.dtypes.value_counts()


# In[ ]:


# Number of unique classes in each object column
train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# # Descriptive Statistics (To detect outliers and anomolies in continuous variables)

# ## Overview of all variables

# In[ ]:


train.describe()


# ## Age

# In[ ]:


(train['DAYS_BIRTH'] / -365).describe()


# ## Amount of time at current job

# In[ ]:


train['DAYS_EMPLOYED'].describe()


# ## Number of Children

# In[ ]:


train['CNT_CHILDREN'].describe()


# ## Income

# In[ ]:


train['AMT_INCOME_TOTAL'].describe()


# ## Number of Days before application that client changed his/her registration

# In[ ]:


train['DAYS_REGISTRATION'].describe()


# ## Preprocessing Notes
# - The max value for the DAYS_EMPLOYED variable appears to be an error.  We should replace this with a NAN value in our preprocessing step.
# - We should replace the DAYS_BIRTH variable with an absolute value, and divide by 365, so it is in years.
# - There are many categorical variables that we need to perform one-hot encoding on in the preprocessing step.
# - We need to perform min-max scaling before feeding the data into machine learning algorithms
# - 67 columns have missing data.  We will need to develop a strategy for dropping columns, rows, or imputing values during the preprocessing step.

# # Distributions of Important Features

# ## Target (repaid loan or not)

# In[ ]:


# TARGET value 0 means loan is repayed, value 1 means loan is not repayed.
plt.figure(figsize=(15,5))
sns.countplot(train.TARGET)
plt.xlabel('Target (0 = repaid, 1 = not repaid)'); plt.ylabel('C'); plt.title('Distribution of Loan Repayment');


# ## Contract Type

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(train.NAME_CONTRACT_TYPE.values,data=train)
plt.xlabel('Contract Type'); plt.ylabel('Count'); plt.title('Distribution of Contract Types');


# ## Gender

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(train.CODE_GENDER.values,data=train)
plt.xlabel('Gender'); plt.ylabel('Number of Clients'); plt.title('Distribution of Gender');


# ## Education Type/Level

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(train.NAME_EDUCATION_TYPE.values,data=train)
plt.xlabel('Education Type/Level'); plt.ylabel('Number of Clients'); plt.title('Distribution of Education Type/Level');


# ## Car Ownership

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(train.FLAG_OWN_CAR.values,data=train)
plt.xlabel('Car Ownership (Y = Yes, N = No)'); plt.ylabel('Number of Clients'); plt.title('Distribution of Car Ownership');


# ## Home Ownership

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(train.FLAG_OWN_REALTY.values,data=train)
plt.xlabel('Home Ownership (Y = Yes, N = No)'); plt.ylabel('Number of Clients'); plt.title('Distribution of Home Ownership');


# ## Number of Children

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(train.CNT_CHILDREN.values,data=train)
plt.xlabel('Number of Children'); plt.ylabel('Number of Clients'); plt.title('Distribution of Children Per Client');


# ## Family Status

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(train.NAME_FAMILY_STATUS.values,data=train)
plt.xlabel('Family Status'); plt.ylabel('Number of Clients'); plt.title('Family Status Distribution');


# ## Housing Type

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(train.NAME_HOUSING_TYPE.values,data=train)
plt.xlabel('Housing Type'); plt.ylabel('Number of Clients'); plt.title('Housing Type Distribution');


# ## Age of Client

# In[ ]:


train['DAYS_BIRTH'] = abs(train['DAYS_BIRTH'])

plt.figure(figsize=(15,5))
sns.distplot(train['DAYS_BIRTH'] / 365,bins=5)
plt.xlabel('Age (Years)'); plt.ylabel('Density'); plt.title('Age Distribution');


# ## Notable Visualization

# In[ ]:


# Age information into a separate dataframe
age_data = train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)


# In[ ]:


# Group by the bin and calculate averages
age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_groups


# In[ ]:


plt.figure(figsize = (8, 8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');


# # Correlations with the Target

# In[ ]:


# Find correlations with the target and sort
correlations = train.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))


# # Preprocessing Notes
# - The max value for the DAYS_EMPLOYED variable appears to be an error.  We should replace this with a NAN value in our preprocessing step.
# - We should replace the DAYS_BIRTH variable with an absolute value, and divide by 365, so it is in years.
# - There are many categorical variables that we need to perform one-hot encoding on in the preprocessing step.
# - We need to perform min-max scaling before feeding the data into machine learning algorithms
# - 67 columns have missing data.  We will need to develop a strategy for dropping columns, rows, or imputing values during the preprocessing step.
# - Many features are highly-correlated with one another.  We will need to remove the unnecessary features.

# # Benchmark Model

# We will use the logistic regression model to establish our benchmark results.  First, we will copy the training and testing sets, so we can perform some basic preprocessing on the data.  Specifically, we will scale the features between 0 and 1, and fill-in all missing values with the median value of the columns, and perform one-hot encoding on categorical variables.

# In[ ]:


# Copy data into a different dataframe to preserve the original
bench_train = train.copy()
bench_test = test.copy()

# one-hot encoding of categorical variables
bench_train = pd.get_dummies(bench_train)
bench_test = pd.get_dummies(bench_test)

# capture the labels
bench_train_labels = bench_train['TARGET']
bench_test_labels = bench_test['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
bench_train, bench_test = bench_train.align(bench_test, join = 'inner', axis = 1)

# Drop the target from the training and testing data
bench_train = bench_train.drop(columns = ['TARGET'])
bench_test = bench_test.drop(columns = ['TARGET'])

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(bench_train)
imputer.fit(bench_test)

# Transform both training and testing data
bench_train = imputer.transform(bench_train)
bench_test = imputer.transform(bench_test)

# Repeat with the scaler
scaler.fit(bench_train)
scaler.fit(bench_test)
bench_train = scaler.transform(bench_train)
bench_test = scaler.transform(bench_test)

print('Training data shape: ', bench_train.shape)
print('Testing data shape: ', bench_test.shape)


# In[ ]:


# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(log_reg, bench_train, bench_train_labels, cv=shuffle, scoring='roc_auc')
print(scores)


# # Data Preprocessing

# ## Steps
# - Replace the max value for the DAYS_EMPLOYED variable with a NAN value.
# - Replace the DAYS_BIRTH variable with an absolute value.
# - Remove any columns that are missing more than 50 percent of their values.
# - Perform one-hot encoding on categorical variables.
# - Perform min-max scaling before feeding the data into machine learning algorithms.
# - Remove colinear features.  If any columns have a correlation over 0.9, only keep one, and remove the others.
# - Use the feature_importances attribute of the lightGBM model to remove features that have very little importance.
# - Impute any additional missing rows with median values.
# 

# ## Copy Data

# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


# Copy data into a different dataframe to preserve the original
main_train = train.copy()
main_test = test.copy()


# ## Handle Outliers and Transformations

# In[ ]:


main_train['DAYS_BIRTH'] = abs(main_train['DAYS_BIRTH'])
main_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)


# ## One-Hot Encode Categorical Features

# In[ ]:


# one-hot encoding of categorical variables
main_train = pd.get_dummies(main_train)
main_test = pd.get_dummies(main_test)

# Align the training and testing data, keep only columns present in both dataframes
main_train, main_test = main_train.align(main_test, join = 'inner', axis = 1)


# In[ ]:


print(main_train.shape)
print(main_test.shape)


# ## Remove Collinear Features Above Threshold

# In[ ]:


# Threshold for removing correlated variables
threshold = 0.8

# Absolute value correlation matrix
corr_matrix = main_train.corr().abs()
corr_matrix.head()


# In[ ]:


upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()


# In[ ]:


# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))


# In[ ]:


main_train = main_train.drop(columns = to_drop)
main_test = main_test.drop(columns = to_drop)

print('Training shape: ', main_train.shape)
print('Testing shape: ', main_test.shape)


# Extra features that have a correlation above 0.8 have now been dropped.

# ## Remove Features with Missing Values Above Threshold

# In[ ]:


# Train missing values (in percent)
train_missing = (main_train.isnull().sum() / len(main_train)).sort_values(ascending = False)
train_missing.head()


# In[ ]:


# Test missing values (in percent)
test_missing = (main_test.isnull().sum() / len(main_test)).sort_values(ascending = False)
test_missing.head()


# In[ ]:


# Identify missing values above threshold
train_missing = train_missing.index[train_missing > 0.50]
test_missing = test_missing.index[test_missing > 0.50]

all_missing = list(set(set(train_missing) | set(test_missing)))
print('There are %d columns with more than 50%% missing values' % len(all_missing))


# In[ ]:


# Need to save the labels because aligning will remove this column
main_train_labels = main_train["TARGET"]
main_train_ids = main_train['SK_ID_CURR']
main_test_ids = main_test['SK_ID_CURR']

main_train = pd.get_dummies(main_train.drop(columns = all_missing))
main_test = pd.get_dummies(main_test.drop(columns = all_missing))

main_train, main_test = main_train.align(main_test, join = 'inner', axis = 1)

print('Training set full shape: ', main_train.shape)
print('Testing set full shape: ' , main_test.shape)


# Columns with greater than 50 percent of observations missing have been removed.

# In[ ]:


main_train = main_train.drop(columns = ['SK_ID_CURR'])
main_test = main_test.drop(columns = ['SK_ID_CURR'])


# In[ ]:


print('Training set full shape: ', main_train.shape)
print('Testing set full shape: ' , main_test.shape)


# ID column is now dropped.  We don't need this extra info when we feed the data to our models.

# ## Impute and Scale Features

# In[ ]:


# capture the labels
main_train_labels = main_train['TARGET']
main_test_labels = main_test['TARGET']

# Drop the target from the training and testing data
main_train = main_train.drop(columns = ['TARGET'])
main_test = main_test.drop(columns = ['TARGET'])

# impute median values
imputer = Imputer(strategy = 'median')
imputer.fit(main_train)
imputer.fit(main_test)
main_train = imputer.transform(main_train)
main_test = imputer.transform(main_test)

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(main_train)
scaler.fit(main_test)
main_train = scaler.transform(main_train)
main_test = scaler.transform(main_test)

print('Training data shape: ', main_train.shape)
print('Testing data shape: ', main_test.shape)


# # Models
# - Logistic Regression
# - K-Nearest Neighbors
# - Naive Bayes
# - SVM Classifier
# - Random Forest
# - Decision Trees
# - AdaBoost
# - XgBoost
# - LightGBM

# ## Logistic Regression

# In[ ]:


log_reg = LogisticRegression(C = 0.0001)

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(log_reg, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


# In[ ]:


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


# ## K Nearest Neighbors

# In[ ]:


knn = KNeighborsClassifier()

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(knn, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


# In[ ]:


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


# ## Naive Bayes

# In[ ]:


nb = GaussianNB()

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(nb, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


# In[ ]:


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


# ## Support Vector Machine

# In[ ]:


svm = SVC()

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(svm, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


# In[ ]:


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


# ## Decision Tree

# In[ ]:


dtc = DecisionTreeClassifier(random_state=0)

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(dtc, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


# In[ ]:


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


# ## Random Forest

# In[ ]:


rfc = RandomForestClassifier()

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(rfc, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


# In[ ]:


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


# ## AdaBoost Classifier

# In[ ]:


abc = AdaBoostClassifier(DecisionTreeClassifier())

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(abc, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


# In[ ]:


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


# ## XGBoost Classifer

# In[ ]:


xgb = XGBClassifier()

shuffle = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(xgb, main_train, main_train_labels, cv=shuffle, scoring='roc_auc')


# In[ ]:


print("All Scores:")
print(scores)
print("Average Score:")
print(round((sum(scores) / len(scores)), 4)) 


# In[ ]:





# In[ ]:





# In[ ]:





# # Refinement

# In[ ]:




