#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

pd.options.display.max_columns = None

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset = pd.read_csv("../input/house-prices-advanced-regression-techniques/housetrain.csv")
dataset = dataset.set_index('Id')


# In[ ]:


dataset.describe(include='all')


# These are a lot of features. Trying to figure out the good and meaningful ones
# 
# Most of the categorical variables have 2-4 unique values and they are highly skwed on 1 value, meaning data is Over and under represented.

# # Correlation Matrix

# In[ ]:


plot = plt.figure(figsize=(19, 15))
plt.matshow(dataset.corr(), fignum=plot.number)
plt.xticks(range(dataset.describe().shape[1]), dataset.describe().columns, fontsize=14, rotation=90)
plt.yticks(range(dataset.describe().shape[1]), dataset.describe().columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title("Correlation Matrix", fontsize=16, pad=100)


# In[ ]:


dataset.corr()


# In[ ]:


#Columns which have good correlation with SalesPrice (target variable)
cols = ['OverallQual', 'YearBuilt', 'TotalBsmtSF', 
        '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'SalePrice']


# **I am not sure of how OverallQual is calculated, but I assume varibles with having suffix/prefix condition/quality are already included in variable "OverallQual". **

# # Useful Insights

# ## 1. Correlation between candidate columns (Checking for Multi-Collinearity)

# In[ ]:


newCorrMat = np.corrcoef(dataset[cols].values.T)
size = plt.figure(figsize=(19,15))
sns.set(font_scale=1.25)
heatM = sns.heatmap(newCorrMat, annot=True, cbar=True, square=True, fmt='0.2f', 
                    xticklabels=cols, yticklabels=cols)
plt.show()


# **Drawing conclusions from above heatmap**
# > **MultiCollinearity**
# >> '1stFlrSF' and 'TotalBsmtSF' are highly correlated with each other, meaning both participate in the result insimilar way. It is a problem of "MultiCollinearity". I'll choose one which is highly correlated with SalePrice i.e. 'TotalBsmtSF'. 
# >> Same goes with 'TotRmsAbvGrd' and 'GrLivArea', I am choosing 'GrLivArea'.
# >> Same goes with 'GarageArea' and 'GarageCars', I am choosing GarageCars
# 
# > 'YearBuilt' is an important attribute, however it is not shown so here.
# > 'OverallQual', 'GrLivArea' are clear winners, as they are highly correlated with SalePrice
# > 'FullBath' does not show any resembleness to me, but data suggest so I am keeping it.
# 
# **Let's plot scatterplot matrix to see how are these values shown on Graph**

# ## 2. Scatter plot of chosen features

# In[ ]:


newCols = ['TotalBsmtSF', 'GrLivArea', 'GarageCars', 'YearBuilt', 'OverallQual', 'FullBath', 'SalePrice']
newDf = dataset[newCols]
sns.pairplot(newDf)


# We'll use plots for detecting outliers.

# In[ ]:


newDf.isna().count()


# **Well, luckily I do not have any missing values**

# ## 3. Check for Outliers

# From the scatter plots above, I am now trying to see if there's any outliers in our data and if there is then remove them

# **Small utility for showing Scatter plot between columns**

# In[ ]:


def scatterPlot(x, y, xlabel, ylabel, title):
    plt.scatter(x=x, y=y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# ### 3.1 TotalBsmtSF vs SalePrice

# In[ ]:


scatterPlot(newDf.TotalBsmtSF, newDf.SalePrice, 'Total Basement Surface area', 'Sales Price', 
           'Total Basement surface area vs Sales Price')


# Okay, so point where x>6000 seems to be little odd. This house suggests that it has largest surface area and is available in very cheap price. Maybe it's a steal deal but I think it will affect my predictions so removing it

# In[ ]:


newDf = newDf.drop(newDf[newDf.TotalBsmtSF > 6000].index)


# ### 3.2 GrLivArea vs SalePrice 

# In[ ]:


scatterPlot(newDf.GrLivArea, newDf.SalePrice, 'Ground living area', 'Sales Price', 
           'Ground living area vs Sales Price')


# Two values with ground area greater than 4000 has price over 600000 which seems reasonable, but one house with living area greater than 4000 has price less than 200000. Seems outlier to me, hence removing this.

# In[ ]:


newDf = newDf.drop(newDf[(newDf.GrLivArea>4000) & (newDf.SalePrice<200000)].index)


# ### 3.3 Garage Car vs SalePrice

# In[ ]:


scatterPlot(newDf.GarageCars, newDf.SalePrice, 'Garage Cars', 'Sales Price', 
           'Garage cars vs Sales Price')


# ### 3.4 YearBuilt vs SalePrice

# In[ ]:


scatterPlot(newDf.YearBuilt, newDf.SalePrice, 'Year built', 'Sales Price', 
           'Year build vs Sales Price')


# ### 3.5 OverallQual vs SalePrice

# In[ ]:


scatterPlot(newDf.OverallQual, newDf.SalePrice, 'Overall Quality', 'Sales Price', 
           'Overall quality vs Sales Price')


# Overall quantity with some value 1 has very low sales price, but that seems okay, as the value is increasing, sales price is also increasing. 

# ### 3.6 FullBath vs SalePrice

# In[ ]:


scatterPlot(newDf.FullBath, newDf.SalePrice, 'Full Bath', 'Sales Price', 
           'Full bath vs Sales Price')


# ## Now that I don't have any outliers left in my Data. Let's see if I need any data transformation

# In[ ]:


sns.distplot(newDf.SalePrice, fit=norm)


# Hmm, Sales price data is skewed and hence needs to be transformed.

# In[ ]:


newDf['SalePrice'] = np.log(newDf.SalePrice)


# In[ ]:


sns.distplot(newDf.GrLivArea, fit=norm)


# In[ ]:


newDf['GrLivArea'] = np.log(newDf.GrLivArea)


# In[ ]:


sns.distplot(newDf.TotalBsmtSF, fit=norm)


# Log transformation of 0 is not possible, but log(1) = 0. So I am putting 1 where TotalBsmtSf is 0, so that my log transform cannot fail.

# In[ ]:


newDf['TotalBsmtSF'] = newDf['TotalBsmtSF'].apply(lambda x: 1 if x==0 else x)
newDf['TotalBsmtSF'] = np.log(newDf['TotalBsmtSF'])


# I am not using any catagorical variable for my model, but if I were to use then we need to transform them to numerical using dummies or one hot encoding

# # Prediction Model

# In[ ]:


X = newDf.iloc[:, :-1]
y = newDf.iloc[:, -1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8, random_state=10)


# In[ ]:


#Random Forest Regression
RF = RandomForestRegressor(n_estimators=10000)
RF.fit(X_train, y_train)
y_predict = RF.predict(X_test)


# In[ ]:


mean_squared_error(y_test, y_predict)


# In[ ]:


boosting = GradientBoostingRegressor(n_estimators=10000, learning_rate=0.01)
boosting.fit(X_train, y_train)
y_predict1 = boosting.predict(X_test)


# In[ ]:


mean_squared_error(y_test, y_predict1)


# In[ ]:




