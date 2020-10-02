#!/usr/bin/env python
# coding: utf-8

# Bike sharing demand prediction

# Import the necessary libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display,HTML
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))


# Exploratery data analysis

# In[ ]:


hour_df=pd.read_csv('../input/hour.csv')
hour_df.head()


# In[ ]:


hour_df.shape


# In[ ]:


hour_df.dtypes


# Rename the attributes for better understanding

# In[ ]:


hour_df.rename(columns={'instant':'rec_id','dteday':'datetime','holiday':'is_holiday','workingday':'is_workingday',
                        'weathersit':'weather_condition','hum':'humidity','mnth':'month',
                        'cnt':'total_count','hr':'hour','yr':'year'},inplace=True)
hour_df.head()


# Describe the dataset

# In[ ]:


hour_df.describe()


# Type casting the datetime and categorical attributes

# In[ ]:


hour_df['datetime']=pd.to_datetime(hour_df.datetime)

hour_df['season']=hour_df.season.astype('category')
hour_df['year']=hour_df.year.astype('category')
hour_df['month']=hour_df.month.astype('category')
hour_df['hour']=hour_df.hour.astype('category')
hour_df['is_holiday']=hour_df.is_holiday.astype('category')
hour_df['weekday']=hour_df.weekday.astype('category')
hour_df['is_workingday']=hour_df.is_workingday.astype('category')
hour_df['weather_condition']=hour_df.weather_condition.astype('category')


# Attributes distribution and trends

# Season wise hourly distribution of counts

# In[ ]:


fig,ax=plt.subplots(figsize=(20,8))
sns.set_style('white')

sns.pointplot(x='hour',y='total_count',data=hour_df[['hour','total_count','season']],hue='season',ax=ax)
ax.set_title('Season wise hourly distribution of counts')
plt.show()
fig,ax1=plt.subplots(figsize=(20,8))
sns.boxplot(x='hour',y='total_count',data=hour_df[['hour','total_count']],ax=ax1)
ax1.set_title('Season wise hourly distribution of counts')
plt.show()


# Weekday wise hourly distribution of counts

# In[ ]:


fig,ax=plt.subplots(figsize=(20,8))
sns.pointplot(x='hour',y='total_count',data=hour_df[['hour','total_count','weekday']],hue='weekday')
ax.set_title('Weekday wise hourly distribution of counts')
plt.show()


# Monthly distribution of counts

# In[ ]:


fig,ax1=plt.subplots(figsize=(20,8))
sns.barplot(x='month',y='total_count',data=hour_df[['month','total_count']],ax=ax1)
ax1.set_title('Monthly distribution of counts')
plt.show()
fig,ax2=plt.subplots(figsize=(20,8))
sns.barplot(x='month',y='total_count',data=hour_df[['month','total_count','season']],hue='season',ax=ax2)
ax2.set_title('Season wise monthly distribution of counts')
plt.show()


# Yearly wise distribution of counts

# In[ ]:


fig,ax=plt.subplots(figsize=(20,8))
sns.violinplot(x='year',y='total_count',data=hour_df[['year','total_count']])
ax.set_title('Yearly wise distribution of counts')
plt.show()
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(20,5))
sns.barplot(data=hour_df,x='is_holiday',y='total_count',hue='season',ax=ax1)
ax1.set_title('is_holiday wise distribution of counts')
sns.barplot(data=hour_df,x='is_workingday',y='total_count',hue='season',ax=ax2)
ax2.set_title('is_workingday wise distribution of counts')
plt.show()


# Outliers distribution

# In[ ]:


fig,ax=plt.subplots(figsize=(20,8))

sns.boxplot(data=hour_df[['temp','windspeed','humidity']])
ax.set_title('temp_windspeed_humidity distribution')
plt.show()


# In[ ]:


fig,(ax1,ax2,ax3)=plt.subplots(nrows=3,figsize=(20,10))
sns.boxplot(x='hour',y='total_count',data=hour_df[['hour','total_count']],ax=ax1)
ax1.set_title('Hourly wise distribution of outliers')

sns.barplot(x='month',y='total_count',data=hour_df[['month','total_count']],ax=ax2)
ax2.set_title('Monthly wise distribution of outliers')

sns.violinplot(x='year',y='total_count',data=hour_df[['year','total_count']],ax=ax3)
ax3.set_title('Yearly wise distribution of outliers')
plt.show()


# Correlation matrix for better understanding between different attributes of the data.

# In[ ]:


correMtr=hour_df[["temp","atemp","humidity","windspeed","total_count"]].corr()
mask=np.array(correMtr)
mask[np.tril_indices_from(mask)]=False
fig,ax=plt.subplots(figsize=(20,5))
sns.heatmap(correMtr,mask=mask,vmax=0.8,square=True,annot=True,ax=ax)
ax.set_title('Correlation matrix of attributes')
plt.show()


# Regression analysis

# In[ ]:


from sklearn import preprocessing,metrics,linear_model
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split


# Split the training dataset

# In[ ]:


X_train,X_test,y_train,y_test= train_test_split(hour_df.iloc[:,0:-3],hour_df['total_count'],test_size=0.3,random_state=42)

X_train=X_train.reset_index() 
y_train=y_train.reset_index()

X_test=X_test.reset_index() 
y_test=y_test.reset_index() 

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
print(y_train.head())
print(y_test.head())


# Normality test(probability plot)

# In[ ]:


import scipy
from scipy import stats
stats.probplot(y_train.total_count.tolist(),dist='norm',plot=plt)
plt.show()


# Split the features into categorical and numerical features

# In[ ]:


training_features=X_train[['season','is_holiday','is_workingday','weather_condition','hour','month','year','weekday','temp','atemp','humidity','windspeed']]
categorical_features=['season','is_holiday','is_workingday','weather_condition']
numerical_features=[['temp','atemp','humidity','windspeed','hour','month','year','weekday']]


# Decoding the training features

# In[ ]:


training_attributes=pd.get_dummies(training_features,columns=categorical_features)
training_attributes.head()


# Linear regression

# In[ ]:


X_train=training_attributes
y_train=y_train.total_count.values
lr=linear_model.LinearRegression()


# fit the training model

# In[ ]:


lr.fit(X_train,y_train)


# Cross validation prediction

# In[ ]:


predict=cross_val_predict(lr,X_train,y_train,cv=3)

fig,ax=plt.subplots(figsize=(20,8))
ax.scatter(y_train,y_train-predict)
ax.axhline(lw=2,color='black')
ax.set_title('Cross validation prediction plot')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()


# From the cross validation prediction plot, we observed that it violates the homoscedasticity assumption
# and it is nothing but if all the random variables sequence or vector have the same finite variance.

# Model evalution metrics

# R-squared and mean squared error scores

# In[ ]:


r2_scores = cross_val_score(lr, X_train, y_train, cv=3)
mse_scores = cross_val_score(lr, X_train, y_train, cv=3,scoring='neg_mean_squared_error')
print(r2_scores)
print(mse_scores) 


# Cross validation scores

# In[ ]:


sns.set_style('whitegrid')
fig,ax=plt.subplots(figsize=(10,5))
ax.plot([i for i in range(len(r2_scores))],r2_scores,lw=2 )
ax.set_xlabel('R-squared')#coefficeint of determination
ax.set_ylabel('Iterated')
ax.set_title('Cross validation scores,Avg:{}'.format(np.average(r2_scores)))
plt.show()


# From the R-squared or coefficient of determination is 0.39 on average for 3-fold cross validation and it means
# that predictor is only able to explain 39% of the variance in the target variable.
# 

# Test data performance

# Split the test dataset to categorical and numerical features

# In[ ]:


test_features= X_test[['season','is_holiday','weather_condition','is_workingday','hour','weekday','month','year','temp','atemp','humidity','windspeed']]
numeric_features = ['temp','humidity','windspeed','hour','weekday','month','year']
test_cat_features =  ['season','is_holiday','weather_condition','is_workingday']


# Decoding the test attributes

# In[ ]:


test_attributes=pd.get_dummies(test_features,columns=test_cat_features)
test_attributes.head()


# fit the model

# In[ ]:


X_test=test_attributes
y_test=y_test.total_count.values
lr.fit(X_test,y_test)


# predict the model

# In[ ]:


y_pred=lr.predict(X_test)


# Model evaluation metrics

# **Root mean squared error and mean absolute error**

# In[ ]:



rmse=sqrt(metrics.mean_squared_error(y_test,y_pred))
print(rmse)
mae=metrics.mean_absolute_error(y_test,y_pred)
print(mae)


# Residual plot

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(y_test, y_test-y_pred)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text("Residual Plot")
plt.show()


# From the linear regression anaysis, we can conclude that model not suitable for this problem due 
# non linearirty of data.

# Decision tree regressor

# In[ ]:


X_train=training_attributes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
dtr=DecisionTreeRegressor(min_samples_split=2,max_leaf_nodes=40)


# fit the model

# In[ ]:


dtr.fit(X_train,y_train)


# Decision tree regression accuracy score

# In[ ]:


dtr.score(X_train,y_train)


# Plot the learned model

# In[ ]:


from sklearn import tree
import pydot
import graphviz

dot_data = tree.export_graphviz(dtr, out_file=None) 
graph = graphviz.Source(dot_data) 
graph


# Randomized search cv with cross validation

# In[ ]:


from scipy.stats import randint as sp_randint
param_random = {"criterion": ["mse", "mae"],
              "min_samples_split": sp_randint(1, 5, 10),
              "max_depth": [2, 6, 8],
              "min_samples_leaf": sp_randint(5, 10,20),
              "max_leaf_nodes": sp_randint( 10, 20, 40),
              }


# Trained the Random Search CV model

# In[ ]:


X_train=training_attributes

randomized_cv_dtr = RandomizedSearchCV(dtr, param_random, cv=3,random_state=32)


# fit the model

# In[ ]:


randomized_cv_dtr.fit(X_train,y_train)


# model best score and best parameters

# In[ ]:


print(randomized_cv_dtr.best_score_)
print(randomized_cv_dtr.best_params_)


# Parameter hypertuning  results

# In[ ]:


df=pd.DataFrame(data=randomized_cv_dtr.cv_results_)
df.head()


# Effect of depth and leaf nodes on model performance

# In[ ]:


fig,ax = plt.subplots()
sns.pointplot(data=df[['mean_test_score',
                           'param_max_leaf_nodes',
                           'param_max_depth']],
             y='mean_test_score',x='param_max_depth',
             hue='param_max_leaf_nodes',ax=ax)
ax.set(title="Effect of Depth and Leaf Nodes on Model Performance")
plt.show()


# Residual plot

# In[ ]:


predicted = randomized_cv_dtr.best_estimator_.predict(X_train)
residuals = y_train.flatten()-predicted
fig, ax = plt.subplots()
ax.scatter(y_train.flatten(), residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()


# R-squared and mean squared error scores

# In[ ]:


r2_scores = cross_val_score(randomized_cv_dtr.best_estimator_, X_train, y_train, cv=3)
print(r2_scores)
mse_scores = cross_val_score(randomized_cv_dtr.best_estimator_, X_train, y_train, cv=3,scoring='neg_mean_squared_error')
print(mse_scores)


# Setting the model for testing

# In[ ]:


best_dtr_model = randomized_cv_dtr.best_estimator_
pred = best_dtr_model.predict(X_test)


# Root mean squared error and mean absolute error

# In[ ]:



rmse=sqrt(metrics.mean_squared_error(y_test,pred))
print(rmse)
mae=metrics.mean_absolute_error(y_test,pred)
print(mae)


# Residual plot

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(y_test.flatten(), y_test.flatten()-pred)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
X_train=training_attributes
rf=RandomForestRegressor(n_estimators=200)


# **Fit the model**

# In[ ]:


rf.fit(X_train,y_train)


# **Random forest accuracy score**

# In[ ]:


rf.score(X_train,y_train)


# **Cross validation prediction**

# In[ ]:


predict=cross_val_predict(rf,X_train,y_train,cv=3)

fig,ax=plt.subplots(figsize=(20,8))
ax.scatter(y_train,y_train-predict)
ax.axhline(lw=2,color='black')
ax.set_title('Cross validation prediction plot')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()


# **R-squared and mean squared error scores**

# In[ ]:


r2_scores = cross_val_score(rf, X_train, y_train, cv=3)
print(r2_scores)
mse_scores = cross_val_score(rf, X_train, y_train, cv=3,scoring='neg_mean_squared_error')
print(mse_scores)


# In[ ]:


sns.set_style('whitegrid')
fig,ax=plt.subplots(figsize=(10,5))
ax.plot([i for i in range(len(r2_scores))],r2_scores,lw=2 )
ax.set_xlabel('R-squared')#coefficeint of determination
ax.set_ylabel('Iterated')
ax.set_title('Cross validation scores,Avg:{}'.format(np.average(r2_scores)))
plt.show()


# **Predict the model**

# In[ ]:


X_test=test_attributes
rf_pred=rf.predict(X_test)
rf_pred


# **Root mean squared error and mean absolute error**

# In[ ]:


rmse=sqrt(metrics.mean_squared_error(y_test,rf_pred))
print(rmse)
mae=metrics.mean_absolute_error(y_test,rf_pred)
print(mae)


# **Residual plot**

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(y_test, y_test-rf_pred)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()


# **Final model for predicting the bike rental count on daily basis**

# When we compare the root mean squared error and mean absolute error of all 3 models, the random forest model has less root mean squared error and mean absolute error. So, finally random forest model is best for predicting the bike rental count on daily basis.

# In[ ]:


Bike_df1=pd.DataFrame(y_test,columns=['y_test'])
Bike_df2=pd.DataFrame(rf_pred,columns=['rf_pred'])
Bike_predictions=pd.merge(Bike_df1,Bike_df2,left_index=True,right_index=True)
Bike_predictions.to_csv('Bike_Rental_Count.csv')
Bike_predictions

