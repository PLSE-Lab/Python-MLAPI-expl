#!/usr/bin/env python
# coding: utf-8

# **----Problem Statement:----**
# 
# A dataset which contains attributes about homes (including sale price) in Ames, Iowa is provided as source. Create a model which could predict the sale price of homes for a "fresh" dataset (with no sale price available).
# 
# 
# **----Approach:----**
# 
# Based on the dataset, it was observed that a multi-variable Linear Regression model would be a good starting point
# 
# **----Solution Steps:----**
# 
# STEP-1 **[Feature selection]**: Out of the available list of 80 features, choose the most impactful list of features in terms of sale price of the house; in order to simplify feature engineering, all categorical attributes, such as "Neighbourhood", were
# not included in the final dataset (this is not realistic in a production environment)
# 
# STEP-2 **[Correlation with Sale Price]**: Create a correlogram displaying the relationship among the attributes from the earlier step. The most useful correlation, however, would be the correlation of house attributes with the sale price-any feature with correlation less then 0.50 (with the sale price) will be dropped.
# 
# STEP-3 **[Build Final Feature List]**: Build a dataset with only features which have a high correlation with the sale price
# 
# STEP-4 **[Data Cleanup]**: Drop the rows with null values or replace them with statistical mean, mode, etc. of the data fields (if meaningful)
# 
# STEP-5 **[Apply Linear Regression]**: Divide the data into test and training sets, so that the model effectiveness can be verified independently. Apply Linear Regression to the cleaned up dataset from step-4 and output the model coefficients and R-squared values for both the test and the training sets 
# 
# *High R-squared values are usually better, since the model would be able to better explain the variances*
# 
# STEP-6 **[Visualize Predictions]**: Create a graph of the predictions 
# 
# Each of the above step is executed below in sequential order
# 
# **----Solution Analysis----**
# 
# By splitting the data into training set for training the model, and a test set for validating the model, the following R-squared values were encountered during **ONE** of the runs.
# 
# R-squared score (training): 0.829
# 
# R-squared score (test): 0.735
# 
# While R-squared alone should not be used for model evaluation, it's a reasonable staring point before employing more sophisticated techniques such as confusion matrices; Furthermore, with the inclusion of categorical attributes such as "Neighborhood" and "House Style", much more accurate model outputs can be achieved
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Create a dataframe with the entire dataset; we will later split this data later during the regression step
house_data = pd.read_csv("../input/train.csv")
house_data.head()


# **STEP-1: Choose the features from the available list**
# 
# *All though not advisable in production scenarios, categorical data columns, such as "Neighborhood" have been removed for ease of computation*

# In[ ]:


#Create dataframe with numeric features
house_data_numeric_fields = house_data[['LotFrontage','LotArea','OverallQual','OverallCond',
                            'YearBuilt','YearRemodAdd','MasVnrArea',
                             'YrSold','MoSold','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','TotRmsAbvGrd',
                             'BedroomAbvGr','FullBath','Fireplaces',
                             'GarageYrBlt','GarageCars','GarageArea','GrLivArea','SalePrice']]

#Display sample rows from above dataframe
house_data_numeric_fields.head()


# In[ ]:





# **STEP-2: Create a correlogram of all the attributes in the numeric data fields of the housing data**
# 
# *This would help in identifying the features which contribute most to the increase or decrease in sale price.
# i.e any correlation value above .5 is statistically significant*

# In[ ]:


#Create correlation dataframe
corr = house_data_numeric_fields.corr()

#Configure figure size
fig, ax = plt.subplots(figsize=(10, 10))

#Produce colour map
colormap = sns.diverging_palette(220, 10, as_cmap=True)

#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")

#Configure x ticks
plt.xticks(range(len(corr.columns)), corr.columns);

#Configure y ticks
plt.yticks(range(len(corr.columns)), corr.columns)

#show plot
plt.show()


# **STEP-3: Build a dataset with only highly correlated values identified in step-2**

# *Based on the above graph, we are able to deduce that the following attributes contribute the most to the price fluctuations (not in any particular order)*
# 
# 1. Ground living Area
# 2. Garage Area
# 3. Number of cars which can be parked in the garage
# 4. The year garage was built
# 5. Fireplace
# 6. Number of full bathrooms
# 7. Total number of rooms
# 8. Overall quality of the house
# 9. The year when the house was built
# 10. Year when last remodelling was done
# 11. Total rooms above ground
# 12. Total basement size in square feet

# In[ ]:



# Create a dataframe with numeric attributes with most correlation with the sale price
house_data_numeric_high_corr = house_data_numeric_fields[['GrLivArea', 'GarageArea','GarageCars','GarageYrBlt','FullBath',
             'TotRmsAbvGrd', 'TotalBsmtSF','YearRemodAdd','YearBuilt','OverallQual','Fireplaces','SalePrice']]


# Visualize the final set of features (from above) to be used to train the LR model, as a scatter matrix. Some interesting observations from the scatter matrix below are as follows:
# 
# **Above ground living area and basement square feet has the most linear relationship to price*
# 
# ** Most homes are less than the half a million mark*
# 
# ** Most of the cheaper houses are old*
# 
# ** Most basements are less than 2500 SF*
# 
# ** Most of the expensive homes were built after 1975*

# In[ ]:


# Plot a scatter matrix

pd.plotting.scatter_matrix(house_data_numeric_high_corr,alpha=0.2, figsize=(20, 20))
plt.show()


# **STEP-4: Get rid of the rows with null values, or use statistical mean, mode, etc. from the attributes as fillers**
# 

# In[ ]:


# identify all columsn which  have null values
house_data_numeric_high_corr_nulls = house_data_numeric_high_corr.loc[:, house_data_numeric_high_corr.isna().any()]

# The year when the garage was built had nulls; it was also observed that the mode of the feature was year was 2005, hence the 
# nulls were replaced by the statistical mode, i.e.year 2005
df_temp = house_data_numeric_high_corr.fillna({'GarageYrBlt':2005})

# All the other features with nulls, except for the garage year built, which was fixed in the above step, were dropped
house_data_numeric_high_corr_cleaned = df_temp.dropna(axis=0,how='any')

# Display sample rows from the cleaned up dataset with no null values
house_data_numeric_high_corr_cleaned.head()


# **STEP-5:Divide the data into test and training sets, so that the model effectiveness can be verified independantly. Apply Linear Regression to the cleaned up dataset from step-4 and output the model coefficeints and R-squared values.**

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# X_true : captures the features in the data set, to which LR will be applied, i.e this the training set
# y_true : captures the sale price in the data set, to which LR will be applied, i.e this the training set

# X_predict : captures the features in the data set, which can be used to verify the model generated using X_true
# y_predict : captures the sale price in the data set, which can be used to verify the model using y_true

linreg = LinearRegression()

# Divide the data into a 30% to 70% split, and use one to train the model, and use the other to predict; hence, training dataset 
# has the sale price; however, sale price has been removed from the dataset to be used for prediction

X_true, X_predict, y_true, y_predict = train_test_split(
                                house_data_numeric_high_corr_cleaned.loc[:, house_data_numeric_high_corr_cleaned.columns != 'SalePrice'],
                                house_data_numeric_high_corr_cleaned['SalePrice'],
                                train_size=.30, test_size=.70)

# Perform curve fitting

linreg = LinearRegression().fit(X_true, y_true)
print('Curve Fitting Complete!')

# Output relevant results from the above step

print('House Prices Dataset')
print('linear model intercept: {}'
     .format(linreg.intercept_))
print('linear model coeff:\n{}'
     .format(linreg.coef_))
print('R-squared score (training): {:.3f}'
     .format(linreg.score(X_true, y_true)))
print('R-squared score (test): {:.3f}'
     .format(linreg.score(X_predict, y_predict)))


# **Predict the test set, based on the curve fitting performed in the above test**

# In[ ]:


predictions = linreg.predict(X_predict)
print('Prediction using the test set based on the curve fitting of the training set is complete!')


# **STEP-6:Graph the predictions**

# In[ ]:


plt.scatter(y_predict,predictions)

