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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix

import os
print(os.listdir("../input"))
import warnings  
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.


# ***Load the Data***
# 
# Let's import the 'kc_house_data.csv' file . I will name the variable as dataset.
# 

# In[ ]:


dataset = pd.read_csv("../input/kc_house_data.csv")


# > ***First 5 rows of data.***
# 
# Since we have loaded the data, now we will read our data. The below will give us the 5 record of our dataset.

# In[ ]:


dataset.head()


# ***Dataset contains:*****
# 
# 
# **Id:** a notation for a house
# 
# **Date:** Date house was sold
# 
# **Price:** Price is prediction target
# 
# **Bedrooms:** Number of Bedrooms/House
# 
# **Bathrooms:** Number of bathrooms/House
# 
# **Sqft_Living:**  square footage of the home
# 
# ***Sqft_Lot:*** square footage of the lot
# 
# ***Floors:*** Total floors (levels) in house
# 
# ***Waterfront:*** House which has a view to a waterfront
# 
# ***View:*** Has been viewed
# 
# ***Condition:*** How good the condition is ( Overall )
# 
# ***Grade:*** overall grade given to the housing unit, based on King County grading system
# 
# ***Sqft_Above:*** square footage of house apart from basement
# 
# ***Sqft_Basement:*** square footage of the basement
# 
# ***Yr_Built:*** Built Year
# 
# ***Yr_Renovated:*** Year when house was renovated
# 
# ***Zipcode:*** Zip
# 
# ***Lat:*** Latitude coordinate
# 
# ***Long:*** Longitude coordinate
# 
# ***Sqft_Living15:*** Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area
# 
# ***Sqft_Lot15:***  lotSize area in 2015(implies-- some renovations)
# 

# ***Null Value Detection***
# Let's Check for null values in the dataset
# 

# In[ ]:


#Check whether there is any null values
dataset.info()


# ***Finding Unique Values:***
# 
# From the above it is clear that the dataset have no null values present, so lets check how many unique values is present for each feature. We will loop through the dataset for checking the unique values present.

# In[ ]:


#Lets find out how many unique values are present in each column

for value in dataset:
    print('For {},{} unique values present'.format(value,dataset[value].nunique()))


# ***Dropping of the particular column value:***
# 
# We don't require the column id and date at this point so we will be dropping them from the dataset.  

# In[ ]:


dataset = dataset.drop(['id','date'],axis=1)


# ***View the modified dataset***
# 
# Let's now again view the dataset using the same head command we used earlier. 

# In[ ]:


dataset.head()


# ***Data Visulaization using seaborn***
# 
# So all the column data remains the same except the two column is dropped. Let's now forward with data visualization using a pairplot
# 

# In[ ]:


plt.figure(figsize=(10,6))
sns.plotting_context('notebook',font_scale=1.2)
g = sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms','grade','yr_built','yr_renovated']]
                 ,hue='bedrooms',size=2)
g.set(xticklabels=[])


# From the above plot it is clear for a linear regression for sqft_living & price, 
# 
# So lets plot them in a joint plot to explore more on the data. 

# In[ ]:



sns.jointplot(x='sqft_lot',y='price',data=dataset,kind='reg',size=4)
sns.jointplot(x='sqft_above',y='price',data=dataset,kind='reg',size=4)
sns.jointplot(x='sqft_living',y='price',data=dataset,kind='reg',size=4)
sns.jointplot(x='yr_built',y='price',data=dataset,kind='reg',size=4)


# In[ ]:


sns.jointplot(x='bedrooms',y='price',data=dataset,kind='scatter',size=4)
sns.jointplot(x='yr_renovated',y='price',data=dataset,kind='scatter',size=4)
sns.jointplot(x='grade',y='price',data=dataset,kind='scatter',size=4)
sns.jointplot(x='sqft_lot',y='sqft_above',data=dataset,kind='scatter',size=4)


# ***Co relation between Variables***
# 
# We wil use heatmap to view the co relation between variables
# 
# 

# In[ ]:


plt.figure(figsize=(15,10))
columns =['price','bedrooms','bathrooms','sqft_living','floors','grade','yr_built','condition']
sns.heatmap(dataset[columns].corr(),annot=True)


# ***Model on the train data***
# 
# We will start building our model using different regression models
# 
# 

# In[ ]:


# X(Independent variables) and y(target variables) 
X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values


# In[ ]:


#Splitting the data into train,test data 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# *** Multiple Linear Regression: ***
# 
# Fitting the train set to multiple linear regression and getting the score of the model

# In[ ]:


mlr = LinearRegression()
mlr.fit(X_train,y_train)
mlr_score = mlr.score(X_test,y_test)
pred_mlr = mlr.predict(X_test)
expl_mlr = explained_variance_score(pred_mlr,y_test)


# ***Decision Tree ***

# In[ ]:


tr_regressor = DecisionTreeRegressor(random_state=0)
tr_regressor.fit(X_train,y_train)
tr_regressor.score(X_test,y_test)
pred_tr = tr_regressor.predict(X_test)
decision_score=tr_regressor.score(X_test,y_test)
expl_tr = explained_variance_score(pred_tr,y_test)


# ***Random Forest Regression Model***

# In[ ]:



rf_regressor = RandomForestRegressor(n_estimators=28,random_state=0)
rf_regressor.fit(X_train,y_train)
rf_regressor.score(X_test,y_test)
rf_pred =rf_regressor.predict(X_test)
rf_score=rf_regressor.score(X_test,y_test)
expl_rf = explained_variance_score(rf_pred,y_test)


# *** Calculate Model Score ***
# 
# Let's calculate the model score to understand how our model performed along with the explained variance score.

# In[ ]:


print("Multiple Linear Regression Model Score is ",round(mlr.score(X_test,y_test)*100))
print("Decision tree  Regression Model Score is ",round(tr_regressor.score(X_test,y_test)*100))
print("Random Forest Regression Model Score is ",round(rf_regressor.score(X_test,y_test)*100))

#Let's have a tabular pandas data frame, for a clear comparison

models_score =pd.DataFrame({'Model':['Multiple Linear Regression','Decision Tree','Random forest Regression'],
                            'Score':[mlr_score,decision_score,rf_score],
                            'Explained Variance Score':[expl_mlr,expl_tr,expl_rf]
                           })
models_score.sort_values(by='Score',ascending=False)


# *** Conclusion***
# 
# From the above it is clear that random forest accuracy is **88%** and also expalined variance score is **0.84**  . So Random Forest is a suitable model for predicting the price of the house. 
# 
# Though there remains other regression model which can bring out the best of the dataset.
# 
# Please upvote if you like my work, this motivates  me to work better :)
# 
# 
# 

# In[ ]:




