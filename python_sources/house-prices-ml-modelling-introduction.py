#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The purpose of the notebook is to provide an introduction into deveopling a regression model using the "House Price: Advanced Regression Techniques" data set. The overall modelling approaches which will be considered in this notebook will include Linear Regression (Lasso, Ridge and Net Elastic), Random Forest (RF) and Gradient Boosting Machine (GBM).
# 
# The key points which will be covered in detail in this notebook include:
# 
# 1. Setting up the Python environment
# 2. Data exploration;
# 3. Handling missing data (including missing data imputation);
# 4. Feature engineering;
# 5. Setting up a validation environent;
# 6. Fitting different regressions including feature engineering;
# 7. Comparing modelling approaches and stacking models.

# # Evaluation
# The first step intacking any modelling problem is to fully understand the target variable you plan to predict. For Kaggle problems in particular, it is equally as important to fully understand the evaluation metric. In order to achieve the best prediction for the Kaggle problem, we will need to set up the validation environment with the evaluation metric in mind.
# 
# Below is the official evaluation metric for this problem:
# 
# _"Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)"_
# 
# The evaluation metric in mathematical notation is therefore:
# 
# \begin{equation}
# RMSE = \sqrt{\frac{\sum_{i=1}^{n}(\hat{y_i} - y_i)}{n}}
# \end{equation}
# 
# where:
# 
# - $y_i$ is the $i^{th}$ actual log of the house sale price, , $\log{(SalePrice)}$;
# - $\hat{y_i}$ is the $i^{th}$ prediction of the log of the house sale price, $\log{(SalePricePred)}$;
# - $n$ is the number of sale prices predicted.

# # 1. Python Environment Setup
# 
# ## 1.1 Importing Python Packages
# Before we begin importing the data we will first import any Python packages which will be useful. Below is a very high level description for each of the packages and how they will be used in this notebook:
# 
# __pandas:__ will provide a data structure and basic data analysis functionality such as merging, grouping and shaping the data.
# 
# __numpy & scipy:__ will be used to handle most of the mathematical operations on the data.
# 
# __seaborn:__ will be used as a framework for plotting data.
# 
# __sklearn:__ will provide the framework for the validation environment and different regression classifiers. All of the regressions in this notebook will leaverage the sklearn package.

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import mean_squared_error
from sklearn.tree import _tree
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


# ## 1.2 Importing Data
# The data that will be used in this notebook is the "House Price: Advanced Regression Techniques" data.

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("Train Shape: {}".format(train.shape))
print("Test Shape: {}".format(test.shape))


# In[ ]:


test = pd.read_csv('../input/test.csv')
print("Test Shape: {}".format(test.shape))


# The "test" dataset has one less colunm because there is no "SalePrice" variable as this needs to be predicted.

# ## 1.3 Target Variable
# The target variable for this problem is the log of the "SalePrice" which is not currently on the "train" dataset. For this reason we will have to add it to the dataset manually. Notice in the plots that LogSalePrice is more normally distributed that SalePrice.

# In[ ]:


train["LogSalePrice"] = np.log1p(train['SalePrice'])

print("Mean(std) of Sale Price: {0:.0f}({1:.0f})".format(train["SalePrice"].mean(), train["SalePrice"].std()))
print("Mean(std) of Log Sale Price: {:.2f}({:.2f})".format(train["LogSalePrice"].mean(), train["LogSalePrice"].std()))

f, axes = plt.subplots(1, 2, figsize=(12,4))
ax1 = sns.distplot(train.SalePrice, ax=axes[0])
ax2 = sns.distplot(train.LogSalePrice, ax=axes[1])
ax1.set(xlabel='Sale Price', ylabel='Proportion', title='Sale Price')
ax2.set(xlabel='Log Sale Price', ylabel='Proportion', title='Log Sale Price')
plt.show()


# # 2. Data Exploration
# This section is dedicated to undetstanding the data and providing a framework for data visualisations. This data visualisation framework will provide some useful tools to visually inspect different data types including numeric and and categorical.
# 
# ## 2.1 Visualising Numeric Data
# A list of numeric variables in the "train" dataset is provided below. Notice that both "SalePrice" and "LogSaePrice" are in this list.

# In[ ]:


print("Numerical Variables:")
train.select_dtypes(exclude=['object']).columns.values


# ### 2.1.1 Area Related Variables
# First type of numeric variable which will be investigated is the area related variables.

# In[ ]:


area_vars = ["LogSalePrice","LotArea","TotalBsmtSF",'1stFlrSF','2ndFlrSF','GrLivArea','GarageArea']
area_train = train[area_vars]
area_train.dropna()
sns.pairplot(area_train)


# ### 2.1.2 Time of Sale and Condition
# The second type of variable that will be considered in the model is the time of the Sale and the condition of the properies. As a result of this analysis we should be able to understand the following:
# 
# - Whether Overall Condition and Quality impacts house price
# - Whether house prices are dependent on the year they were sold
# - Whether there is any seasonality in sale price

# In[ ]:


f, axes = plt.subplots(1, 2, figsize=(12,4))
ax1 = sns.violinplot(x="OverallQual", y="LogSalePrice", data=train, ax=axes[0])
ax2 = sns.violinplot(x="OverallCond", y="LogSalePrice", data=train, ax=axes[1])
ax1.set(xlabel='Quality', ylabel='Log Sale Price', title='Overall Quality')
ax2.set(xlabel='Condition', ylabel='Log Sale Price', title='Overall Condition')
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
ax = sns.boxplot(x="YearBuilt", y="LogSalePrice", data=train)
ax.set_xlabel(xlabel='Year Sold')
ax.set_ylabel(ylabel='Log Sale Price')
ax.set_title(label='House Price by Year Sold')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
ax = sns.boxplot(x="YearRemodAdd", y="LogSalePrice", data=train)
ax.set_xlabel(xlabel='Year Renovated')
ax.set_ylabel(ylabel='Log Sale Price')
ax.set_title(label='House Price by Year Renovated')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(6,4))
ax = sns.boxplot(x="MoSold", y="LogSalePrice", data=train)
ax.set_xlabel(xlabel='Sale Month')
ax.set_ylabel(ylabel='Log Sale Price')
ax.set_title(label='Month of Sale')
plt.show()


# The key points to take away from this analysis are:
# 
# - Overall quality is more predictive of price than condition. However, both of these variables seem to be relitively predictive
# - Higher quality houses tend to be built more recently
# - Only a slight increase in house price with time. This could be a result of higher wuality houses being built
# - No seasonality effect in house price

# In[ ]:


print("Categorical Variables:")
train.select_dtypes(include=['object']).columns.values


# In[ ]:


# Function to plot categical data against the target variable "SalePrice"
def category_boxplot(table, var):
    grouped = table.groupby(var)['SalePrice'].mean().sort_values(ascending=False)
    sns.boxplot(x=var, y='LogSalePrice', data=table, order=grouped.index)
    
category_boxplot(train, "SaleCondition")


# ### Converting Categorical Numeric Data
# By investigating each of the numeric variables in the above way, two numeric variables were identified as ordinal and one as categorical. For simplicity we will treat these as categorical for now by converting to a string.

# In[ ]:


train['MSSubClass'] = train['MSSubClass'].apply(str)
train['OverallCond'] = train['OverallCond'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)


# ## Missing Data 
# Understanding missing data is another important step. Where variables have missing information, we will have to either drop the variables or consider imputing the missing values. The chart below shows the percentage of missing values in each of the variables.

# In[ ]:


# Function to plot missing data percentage
def missing_plot(table):
    f, ax = plt.subplots()
    plt.xticks(rotation='90')
    sns.barplot(x=table.index, y=table)
    plt.xlabel('Features')
    plt.ylabel('Percentage of Missing Values')
    plt.title('Percentage of Missing Data by Features')

train_miss = (train.isnull().sum() / len(train)) * 100
train_miss = train_miss.drop(train_miss[train_miss == 0].index).sort_values(ascending=False)
missing_plot(train_miss)


# For simplicity, let us consider a threashold of 50%. Therefore any variables with more than 50% missing values will be excluded from the data and variables with less than 50% missing will be imputed.
# 

# In[ ]:


train_drop = train_miss.drop(train_miss[train_miss < 50].index).index.values
train_drop


# In[ ]:


print("Pre Drop Shape: {}".format(train.shape))

for i, col in enumerate(train_drop):
    if i==1:
        train2 = train.drop(col, axis=1)
    elif i>1:
        train2 = train2.drop(col, axis=1)
    
print("Post Drop Shape: {}".format(train2.shape))


# ### Impute Missing Data 
# After dropping all of the variable where more than 50% of values are missing, we can not impute the remaining variables with missing values using a basic group average approach. Note that we could also consider more advanced imputation approaches (e.g. KNN), however the objective of this notebook is to understand the difference in modelling techniques on the same data. Optimising the imputation is therefore not a key priority.

# In[ ]:


cats = train2.select_dtypes(include=['object'])
cols = cats.columns.values
df2=pd.DataFrame([0], columns=['count'], index=['Test'])

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(cats[c].values))
    cats[c] = lbl.transform(list(cats[c].values))
    df = pd.DataFrame([len(cats[c].unique())], columns=['count'], index=[c])
    df2 = df2.append(df)
df2 = df2.drop(df2[df2['count'] == 0].index)
df2 = df2.sort_values(by=['count'], ascending=False)

