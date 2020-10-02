#!/usr/bin/env python
# coding: utf-8

# # ****California Housing Prices (a linear regression and a XGboost)****
# 
# ## Table of Contents:
# * [1-Preprocessing the data](#preprocessing)
# * [2-Linear Regression](#Regression)
#     * [2.1-Training the model](#Training)
#     * [2.2-Evaluating the model](#Evaluation)
# * [3-XGBoost](#Xgboost)
#     * [3.1-Training the model](#Training2)
#     * [3.2-Evaluating the model](#Evaluation2)
# * [4-XGBoost vs Linear Regression](#Comparison)

# ## Preprocessing the data <a class="anchor" id="preprocessing"></a>

# In[ ]:


#importing the libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#loading the dataset and obtaining info about columns
df=pd.read_csv("../input/housing.csv")
list(df)


# In[ ]:


#description of the numerical columns
df.describe()


# In[ ]:


#count the values of the columns
df.count()


# In[ ]:


#We have missing values in the column total_bedrooms. We can drop the null rows or replace the null value for the mean.
#I choose to replace it with the mean
df['total_bedrooms'].fillna(df['total_bedrooms'].mean(), inplace=True)


# In[ ]:


#I want information about the column "ocean_proximity"
df['ocean_proximity'].value_counts()


# In[ ]:


#Transform the variable into a numerical one.
def map_age(age):
    if age == '<1H OCEAN':
        return 0
    elif age == 'INLAND':
        return 1
    elif age == 'NEAR OCEAN':
        return 2
    elif age == 'NEAR BAY':
        return 3
    elif age == 'ISLAND':
        return 4
df['ocean_proximity'] = df['ocean_proximity'].apply(map_age)


# In[ ]:


#Obtaining info of the correlations with a heatmap
plt.figure(figsize=(15,8))
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), linewidths=.5,annot=True,mask=mask,cmap='coolwarm')


# In[ ]:


#There is a high correlation between households and populati
df.drop('households', axis=1, inplace=True)


# In[ ]:


# let's create 2 more columns with the total bedrooms and rooms per population in the same block.
df['average_rooms']=df['total_rooms']/df['population']
df['average_bedrooms']=df['total_bedrooms']/df['population']


# In[ ]:


#dropping the 2 columns we are not going to use
df.drop('total_rooms',axis=1,inplace=True)
df.drop('total_bedrooms',axis=1,inplace=True)


# In[ ]:


#Obtaining info of the new correlations with a heatmap
plt.figure(figsize=(15,8))
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), linewidths=.5,annot=True,mask=mask,cmap='coolwarm')


# In[ ]:


#histogram to get the distributions of the different variables
df.hist(bins=70, figsize=(20,20))
plt.show()


# In[ ]:


#Finding Outliers
plt.figure(figsize=(15,5))
sns.boxplot(x=df['housing_median_age'])
plt.figure()
plt.figure(figsize=(15,5))
sns.boxplot(x=df['median_house_value'])


# In[ ]:


#removing outliers
df=df.loc[df['median_house_value']<500001,:]


# # Linear Regression <a class="anchor" id="Regression"></a>

# ## Training the model <a class="anchor" id="Training"></a>

# In[ ]:


#Choosing the dependant variable and the regressors. In this case we want to predict the housing price
X=df[['longitude',
 'latitude',
 'housing_median_age',
 'population',
 'median_income',
 'ocean_proximity',
 'average_rooms',
 'average_bedrooms']]
Y=df['median_house_value']


# In[ ]:


#splitting the dataset into the train set and the test set

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)


# In[ ]:


#Training the model
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,Y_train)


# In[ ]:


#Obtaining the predictions
y_pred = regressor.predict(X_test)


# ## Evaluating the model <a class="anchor" id="Evaluation"></a>

# In[ ]:


#R2 score
from sklearn.metrics import r2_score
r2=r2_score(Y_test,y_pred)
print('the R squared of the linear regression is:', r2)


# In[ ]:


#Graphically
grp = pd.DataFrame({'prediction':y_pred,'Actual':Y_test})
grp = grp.reset_index()
grp = grp.drop(['index'],axis=1)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,10))
plt.plot(grp[:120],linewidth=2)
plt.legend(['Actual','Predicted'],prop={'size': 20})


# # XGBoost <a class="anchor" id="Xgboost"></a>

# ## Training the model <a class="anchor" id="Training2"></a>

# In[ ]:


import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 1,eta=0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 2000)


# In[ ]:


xg_reg.fit(X_train,Y_train)

y_pred2 = xg_reg.predict(X_test)


# ## Evaluating the model <a class="anchor" id="Evaluation2"></a>

# In[ ]:


#Graphically
grp = pd.DataFrame({'prediction':y_pred2,'Actual':Y_test})
grp = grp.reset_index()
grp = grp.drop(['index'],axis=1)
plt.figure(figsize=(20,10))
plt.plot(grp[:120],linewidth=2)
plt.legend(['Actual','Predicted'],prop={'size': 20})


# In[ ]:


r2xgb=r2_score(Y_test,y_pred2)
print('the R squared of the xgboost method is:', r2xgb)


# In[ ]:


xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


# In[ ]:


#Doing cross validation to see the accuracy of the XGboost model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(regressor, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# # Linear regression vs XGBoost <a class="anchor" id="Comparison"></a>

# In[ ]:


#comparing the scores of both techniques 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

mae1 = mean_absolute_error(Y_test, y_pred)
rms1 = sqrt(mean_squared_error(Y_test, y_pred))
mae2 =mean_absolute_error(Y_test,y_pred2)
rms2 = sqrt(mean_squared_error(Y_test, y_pred2))

print('Stats for the linea regression: \n','mean squared error: ',rms1, '\n R2:',r2,' \n mean absolute error:',mae1 )
print('Stats xgboost: \n','mean squared error: ',rms2, '\n R2:',r2xgb,' \n mean absolute error:',mae2 )


# In[ ]:




