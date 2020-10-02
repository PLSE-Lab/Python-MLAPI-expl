#!/usr/bin/env python
# coding: utf-8

# **Project title :- Bike Renting using Python**

# **Problem statement :-**

# The objective of this Case is to Predication of bike rental count on daily based on the environmental and seasonal settings.

# **Contents :-**
#              
#         1. Exploratory Data Analysis
#            * Loading the dataset and libraries
#            * Data cleaning
#            * Typecasting the attributes
#            * Missing value analysis
#         2. Attributes distributions and trends
#            * Monthly distribution of counts
#            * Yearly distribution of counts
#            * Outliers analysis
#         3. Normality test
#         4. Correlation matrix 
#         5. Split the dataset into train and test dataset
#         6. Encoding the categorical features
#         7. Modelling the training dataset
#            * Linear Regression Model
#            * Decision Tree Regressor Model
#            * Random Forest Model
#         8. Cross Validation Prediction
#            * Linear Regression CV Prediction
#            * Decision Tree Regressor CV Prediction
#            * Random Forest CV Prediction
#         9. Model performance on test dataset
#            * Linear Regression Prediction
#            * Decision Tree Regressor Prediction
#            * Random Forest Prediction
#         10. Model Evaluation Metrics
#            * R-squared score
#            * Root mean square error
#            * Mean absolute error
#         11.Choosing best model for predicting bike rental count

# **Exploratory Data Analysis**

# **Import the required libraries**

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))


# **Read the training data**

# In[ ]:


#import the csv file
bike_df=pd.read_csv("../input/day.csv")


# **Shape of the dataset**

# In[ ]:


#Shape of the dataset
bike_df.shape


# The dataset contains 731 observations and 16 attributes.

# **Data types**

# In[ ]:


#Data types
bike_df.dtypes


# In[ ]:


#Read the data
bike_df.head(5)


# **Rename the columns for better understanding of variables**

# In[ ]:


#Rename the columns
bike_df.rename(columns={'instant':'rec_id','dteday':'datetime','yr':'year','mnth':'month','weathersit':'weather_condition',
                       'hum':'humidity','cnt':'total_count'},inplace=True)


# In[ ]:


#Read the data
bike_df.head(5)


# **Typecasting the datetime and numerical attributes**

# In[ ]:


#Type casting the datetime and numerical attributes to category

bike_df['datetime']=pd.to_datetime(bike_df.datetime)

bike_df['season']=bike_df.season.astype('category')
bike_df['year']=bike_df.year.astype('category')
bike_df['month']=bike_df.month.astype('category')
bike_df['holiday']=bike_df.holiday.astype('category')
bike_df['weekday']=bike_df.weekday.astype('category')
bike_df['workingday']=bike_df.workingday.astype('category')
bike_df['weather_condition']=bike_df.weather_condition.astype('category')


# In[ ]:


#Summary of the dataset
bike_df.describe()


# **Missing value analysis**

# No missing values present in training dataset.

# In[ ]:


#Missing values in dataset
bike_df.isnull().sum()


# **Attributes distributions and trends**

# **Monthly distribution of counts**

# In[ ]:


fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')
#Bar plot for seasonwise monthly distribution of counts
sns.barplot(x='month',y='total_count',data=bike_df[['month','total_count','season']],hue='season',ax=ax)
ax.set_title('Seasonwise monthly distribution of counts')
plt.show()
#Bar plot for weekday wise monthly distribution of counts
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='month',y='total_count',data=bike_df[['month','total_count','weekday']],hue='weekday',ax=ax1)
ax1.set_title('Weekday wise monthly distribution of counts')
plt.show()


# From the above plots, we can observed that increasing the bike rental count in springe and summer season and then decreasing the bike rental count in fall and winter season.
# Here, 
# 
# season 1-> spring season 2 -> summer season 3 -> fall season 4 -> winter

# **Yearly wise distribution of counts**

# In[ ]:


fig,ax=plt.subplots(figsize=(15,8))
#Violin plot for yearly distribution of counts
sns.violinplot(x='year',y='total_count',data=bike_df[['year','total_count']])
ax.set_title('Yearly distribution of counts')
plt.show()


# From the violin plot, we can observed that the bike rental count distribution is highest in year 2012 then in year 2011. 
# 
# Here,  
# year 0-> 2011, year 1-> 2012

# **Holiday wise distribution of counts**

# In[ ]:


fig,ax=plt.subplots(figsize=(15,8))
#Barplot for Holiday distribution of counts
sns.barplot(data=bike_df,x='holiday',y='total_count',hue='season')
ax.set_title('Holiday wise distribution of counts')
plt.show()


# From the above bar plot, we can observed that during no holiday the bike rental counts is highest compared to during holiday for different seasons.
# 
# Here, 0->No holiday, 1-> holiday

# **Workingday wise distribution of counts**

# In[ ]:


fig,ax=plt.subplots(figsize=(15,8))
#Bar plot for workingday distribution of counts
sns.barplot(data=bike_df,x='workingday',y='total_count',hue='season')
ax.set_title('Workingday wise distribution of counts')
plt.show()


# From the above bar plot, we can observed that during workingday the bike rental counts is quite highest compared to during no workingday for different seasons.
# 
# Here, 0-> No workingday, 1-> workingday

# **Weather_condition distribution of counts**

# In[ ]:


fig,ax1=plt.subplots(figsize=(15,8))
#Bar plot for weather_condition distribution of counts
sns.barplot(x='weather_condition',y='total_count',data=bike_df[['month','total_count','weather_condition']],ax=ax1)
ax1.set_title('Weather_condition wise monthly distribution of counts')
plt.show()


# From the above bar plot, we can observed that during clear,partly cloudy weather the bike rental count is highest and the second highest is during mist cloudy weather and followed by third highest during light snow and light rain weather.

# **Outlier analysis**

# **Total_Count_Outliers**

# In[ ]:


fig,ax=plt.subplots(figsize=(15,8))
#Boxplot for total_count outliers
sns.boxplot(data=bike_df[['total_count']])
ax.set_title('total_count outliers')
plt.show()


# From the box plot, we can observed that no outliers are present in total_count variable.

# **Temp_windspeed_humidity_outliers**

# In[ ]:


fig,ax=plt.subplots(figsize=(15,8))
#Box plot for Temp_windspeed_humidity_outliers
sns.boxplot(data=bike_df[['temp','windspeed','humidity']])
ax.set_title('Temp_windspeed_humidity_outiers')
plt.show()


# From the box plot, we can observed that no outliers are present in normalized temp but few outliers are present in normalized windspeed and humidity variable.

# **Replace and impute the outliers**

# In[ ]:


from fancyimpute import KNN

#create dataframe for outliers
wind_hum=pd.DataFrame(bike_df,columns=['windspeed','humidity'])
 #Cnames for outliers                     
cnames=['windspeed','humidity']       
                      
for i in cnames:
    q75,q25=np.percentile(wind_hum.loc[:,i],[75,25]) # Divide data into 75%quantile and 25%quantile.
    iqr=q75-q25 #Inter quantile range
    min=q25-(iqr*1.5) #inner fence
    max=q75+(iqr*1.5) #outer fence
    wind_hum.loc[wind_hum.loc[:,i]<min,:i]=np.nan  #Replace with NA
    wind_hum.loc[wind_hum.loc[:,i]>max,:i]=np.nan  #Replace with NA
#Imputating the outliers by mean Imputation
wind_hum['windspeed']=wind_hum['windspeed'].fillna(wind_hum['windspeed'].mean())
wind_hum['humidity']=wind_hum['humidity'].fillna(wind_hum['humidity'].mean())


# **Replace the original dataset to imputated data**

# In[ ]:


#Replacing the imputated windspeed
bike_df['windspeed']=bike_df['windspeed'].replace(wind_hum['windspeed'])
#Replacing the imputated humidity
bike_df['humidity']=bike_df['humidity'].replace(wind_hum['humidity'])
bike_df.head(5)


# **Normal Probability Plot**

# Normal probability plot is a graphical technique to identify substantive departures from normality and also it tells about goodness of fit.

# In[ ]:


import scipy
from scipy import stats
#Normal plot
fig=plt.figure(figsize=(15,8))
stats.probplot(bike_df.total_count.tolist(),dist='norm',plot=plt)
plt.show()


# The above probability plot, the some target variable data points are deviates from normality.

# **Correlation matrix**

# Correlation matrix is tells about linear relationship between attributes and help us to build better models.

# In[ ]:


#Create the correlation matrix
correMtr=bike_df[["temp","atemp","humidity","windspeed","casual","registered","total_count"]].corr()
mask=np.array(correMtr)
mask[np.tril_indices_from(mask)]=False
#Heat map for correlation matrix of attributes
fig,ax=plt.subplots(figsize=(15,8))
sns.heatmap(correMtr,mask=mask,vmax=0.8,square=True,annot=True,ax=ax)
ax.set_title('Correlation matrix of attributes')
plt.show()


# From correlation plot, we can observed that some features are positively correlated or some are negatively correlated to each other. The temp and atemp are highly positively correlated to each other, it means that both are carrying same information.The total_count,casual and registered are highly positively correlated to each other. So, we are going to ignore atemp,casual and registered variable for further analysis.

# **Modelling the dataset**

# In[ ]:


#load the required libraries
from sklearn import preprocessing,metrics,linear_model
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split


# Split the dataset into train and test in the ratio of 70:30

# In[ ]:


#Split the dataset into the train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(bike_df.iloc[:,0:-3],bike_df.iloc[:,-1],test_size=0.3, random_state=42)

#Reset train index values
X_train.reset_index(inplace=True)
y_train=y_train.reset_index()

# Reset train index values
X_test.reset_index(inplace=True)
y_test=y_test.reset_index()

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
print(y_train.head())
print(y_test.head())


# **Split the features into categorical and numerical features**

# In[ ]:


#Create a new dataset for train attributes
train_attributes=X_train[['season','month','year','weekday','holiday','workingday','weather_condition','humidity','temp','windspeed']]
#Create a new dataset for test attributes
test_attributes=X_test[['season','month','year','weekday','holiday','workingday','humidity','temp','windspeed','weather_condition']]
#categorical attributes
cat_attributes=['season','holiday','workingday','weather_condition','year']
#numerical attributes
num_attributes=['temp','windspeed','humidity','month','weekday']


# **Decoding the training attributes**

# In[ ]:


#To get dummy variables to encode the categorical features to numeric
train_encoded_attributes=pd.get_dummies(train_attributes,columns=cat_attributes)
print('Shape of transfomed dataframe::',train_encoded_attributes.shape)
train_encoded_attributes.head(5)


# **Training dataset**

# In[ ]:


#Training dataset for modelling
X_train=train_encoded_attributes
y_train=y_train.total_count.values


# **Linear Regression Model**

# In[ ]:


#training model
lr_model=linear_model.LinearRegression()
lr_model


# **fit the training model**

# In[ ]:


#fit the trained model
lr_model.fit(X_train,y_train)


# **Accuracy of model**

# In[ ]:


#Accuracy of the model
lr=lr_model.score(X_train,y_train)
print('Accuracy of the model :',lr)
print('Model coefficients :',lr_model.coef_)
print('Model intercept value :',lr_model.intercept_)


# **Cross validation prediction**

# In[ ]:


#Cross validation prediction
predict=cross_val_predict(lr_model,X_train,y_train,cv=3)
predict


# **Cross validation prediction plot**

# In[ ]:


#Cross validation plot
fig,ax=plt.subplots(figsize=(15,8))
ax.scatter(y_train,y_train-predict)
ax.axhline(lw=2,color='black')
ax.set_title('Cross validation prediction plot')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()


# Cross validation prediction plot tells about finite variance between actual target value and predicted target value. In this plot, some data points are have same finite variance between them and for some are not have it.

# **Model evalution metrics**

# **R-squared and mean squared error score**

# In[ ]:


#R-squared scores
r2_scores = cross_val_score(lr_model, X_train, y_train, cv=3)
print('R-squared scores :',np.average(r2_scores))


# The R-squared or coefficient of determination is 0.80 on average for 3-fold cross validation , it means that predictor is only able to predict 80% of the variance in the target variable which is contributed by independent variables.

# **Decoding the test attributes**

# In[ ]:


#To get dummy variables to encode the categorical features to numeric
test_encoded_attributes=pd.get_dummies(test_attributes,columns=cat_attributes)
print('Shape of transformed dataframe :',test_encoded_attributes.shape)
test_encoded_attributes.head(5)


# **Model performance on test dataset**

# In[ ]:


#Test dataset for prediction
X_test=test_encoded_attributes
y_test=y_test.total_count.values


# **Predict the model**

# In[ ]:


#predict the model
lr_pred=lr_model.predict(X_test)
lr_pred


# **Model evaluation metrics**

# **Root mean square error and mean absolute error scores**

# In[ ]:


import math
#Root mean square error 
rmse=math.sqrt(metrics.mean_squared_error(y_test,lr_pred))
#Mean absolute error
mae=metrics.mean_absolute_error(y_test,lr_pred)
print('Root mean square error :',rmse)
print('Mean absolute error :',mae)


# **Residual plot**

# In[ ]:


#Residual plot
fig, ax = plt.subplots(figsize=(15,8))
ax.scatter(y_test, y_test-lr_pred)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text("Residual Plot")
plt.show()


# Residual plot tells about finite variance between actual target value and predicted target value.In this plot,very less data points are have same finite variance between them and for most are not have it.

# **Decision tree regressor**

# In[ ]:


#training the model
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(min_samples_split=2,max_leaf_nodes=10)


# **Fit the model**

# In[ ]:


#Fit the trained model
dtr.fit(X_train,y_train)


# **Decision tree regression accuracy score**

# In[ ]:


#Accuracy score of the model
dtr_score=dtr.score(X_train,y_train)
print('Accuracy of model :',dtr_score)


# **Plot the learned model**

# In[ ]:


#Plot the learned model
from sklearn import tree
import pydot
import graphviz

# export the learned model to tree
dot_data = tree.export_graphviz(dtr, out_file=None) 
graph = graphviz.Source(dot_data) 
graph


# Cross validation prediction

# In[ ]:


predict=cross_val_predict(dtr,X_train,y_train,cv=3)
predict


# **Cross validation prediction plot**

# In[ ]:


# Cross validation prediction plot
fig,ax=plt.subplots(figsize=(15,8))
ax.scatter(y_train,y_train-predict)
ax.axhline(lw=2,color='black')
ax.set_title('Cross validation prediction plot')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()


# Cross validation prediction plot tells about finite variance between actual target value and predicted target value. In this plot,some data points are have same finite variance between them and for some are not have it.

# **Model evalution metrics**

# **R-squared and mean squared error scores**

# In[ ]:


#R-squared scores
r2_scores = cross_val_score(dtr, X_train, y_train, cv=3)
print('R-squared scores :',np.average(r2_scores))


# The R-squared or coefficient of determination is 0.74 on average for 3-fold cross validation ,it means that predictor is only able to predict 74% of the variance in the target variable which is contributed by independent variables.

# **Model performance on test dataset**

# In[ ]:


#predict the model
dtr_pred=dtr.predict(X_test)
dtr_pred


# **Root mean squared error and mean absolute error**

# In[ ]:


#Root mean square error
rmse=math.sqrt(metrics.mean_squared_error(y_test,dtr_pred))
#Mean absolute error
mae=metrics.mean_absolute_error(y_test,dtr_pred)
print('Root mean square error :',rmse)
print('Mean absolute error :',mae)


# **Residual plot**

# In[ ]:


#Residual scatter plot
residuals = y_test-dtr_pred
fig, ax = plt.subplots(figsize=(15,8))
ax.scatter(y_test, residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
ax.set_title('Residual plot')
plt.show()


# Residual plot tells about finite variance between actual target value and predicted target value. In this plot, some data points are have same finite variance between them and for some are not have it.

# **Random Forest**

# In[ ]:


#Training the model
from sklearn.ensemble import RandomForestRegressor
X_train=train_encoded_attributes
rf=RandomForestRegressor(n_estimators=200)


# **Fit the model**

# In[ ]:


#Fit the trained model
rf.fit(X_train,y_train)


# **Random forest accuracy score**

# In[ ]:


#accuracy of the model
rf_score =rf.score(X_train,y_train)
print('Accuracy of the model :',rf_score)


# **Cross validation prediction**

# In[ ]:


#Cross validation prediction
predict=cross_val_predict(rf,X_train,y_train,cv=3)
predict


# **Cross validation prediction plot**

# In[ ]:


#Cross validation prediction plot
fig,ax=plt.subplots(figsize=(15,8))
ax.scatter(y_train,y_train-predict)
ax.axhline(lw=2,color='black')
ax.set_title('Cross validation prediction plot')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()


# Cross validation prediction plot tells about finite variance between actual target value and predicted target value. In this plot,some data points are have same finite variance between them and for some are not have it.

# **R-squared and mean squared error scores**

# In[ ]:


#R-squared scores
r2_scores = cross_val_score(rf, X_train, y_train, cv=3)
print('R-squared scores :',np.average(r2_scores))


# The R-squared or coefficient of determination is 0.85 on average for 3-fold cross validation , it means that predictor is only able to predict 85% of the variance in the target variable which is contributed by independent variables.

# **Model performance on test dataset**

# In[ ]:


#predict the model
X_test=test_encoded_attributes
rf_pred=rf.predict(X_test)
rf_pred


# **Root mean squared error and mean absolute error**

# In[ ]:


#Root mean square error
rmse = math.sqrt(metrics.mean_squared_error(y_test,rf_pred))
print('Root mean square error :',rmse)
#Mean absolute error
mae=metrics.mean_absolute_error(y_test,rf_pred)
print('Mean absolute error :',mae)


# **Residual plot**

# In[ ]:


#Residual scatter plot
fig, ax = plt.subplots(figsize=(15,8))
residuals=y_test-rf_pred
ax.scatter(y_test, residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.set_title('Residual plot')
plt.show()


# Cross validation prediction plot tells about finite variance between actual target value and predicted target value.In this plot,some data points are have same finite variance between them and for some are not have it.

# **Final model for predicting the bike rental count on daily basis**

# When we compare the root mean squared error and mean absolute error of all 3 models, the random forest model has less root mean squared error and mean absolute error. So, finally random forest model is bset for predicting the bike rental count on daily basis.

# In[ ]:


Bike_df1=pd.DataFrame(y_test,columns=['y_test'])
Bike_df2=pd.DataFrame(rf_pred,columns=['rf_pred'])
Bike_predictions=pd.merge(Bike_df1,Bike_df2,left_index=True,right_index=True)
Bike_predictions.to_csv('Bike_Renting_Python.csv')
Bike_predictions

