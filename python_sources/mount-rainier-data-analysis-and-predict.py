#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# #### we are to work on data climbing and weather , data was collected for the period between 2014 and 2015 , for some period we have data for climbing not for weather and the opposite for other period as we will see later , our object here is to study the trend off attempts , and styudy the effect of the weather on success percentage , so we can predict number od attempts and success percentage in the future.
# 
# #### for that purpose we have to get the summation of attempts and successes for each day , then get the weather data for these days to study how attempts vary , and study the effect of weather on successsful percentage.

# ### First : we load necessary libraries 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
import statsmodels.api as sm
from pylab import rcParams
import warnings
import itertools
from sklearn.linear_model import LinearRegression


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# > ### Second : We load the data 

# In[ ]:



# Any results you write to the current directory are saved as output.

data_climb = pd.read_csv('../input/mount-rainier-weather-and-climbing-data/climbing_statistics.csv')
data_wheather = pd.read_csv('../input/mount-rainier-weather-and-climbing-data/Rainier_Weather.csv')


# #### then we explore it to make sure of appearence about missing values 

# In[ ]:


data_climb.head()


# In[ ]:


data_wheather.head()


# In[ ]:


data_climb.info()


# In[ ]:


data_wheather.info()


# #### we notice that both datasets contain no missing value but also date columns are read as strings that should be fixed
# #### we notice also that there is more than a record for a single date in climb data , we just need to know number of unique dates

# In[ ]:


data_climb.Date.drop_duplicates().count()


# #### to process date columns we need to transform them from string to be read as date time columns

# In[ ]:


data_wheather['Date'] = pd.to_datetime(data_wheather['Date'])


# In[ ]:


data_wheather.info()


# In[ ]:


data_wheather.head()


# In[ ]:


data_wheather['Date'].max()


# In[ ]:


data_climb['Date'] = pd.to_datetime(data_climb['Date'])


# In[ ]:


data_climb.info()


# ### Third : Aggregating climb data per date :

# #### since we have multiple attempts per day , we need to get the summation of attempts and successes per day , and about route , as a categorical column , we get the mode , the most common route per day as follows :

# In[ ]:


data_climb.head()


# In[ ]:


s=data_climb.groupby(['Date'])[ 'Attempted' , 'Succeeded' , 'Success Percentage'].agg({'Attempted' :'sum' ,
                                                                                       'Succeeded' :'sum' ,
                                                                                        'Success Percentage':'mean' ,
                                                                                         'Route' : lambda x:x.value_counts().index[0]}).sort_values(by = 'Attempted' , ascending=False)


# In[ ]:


s


# #### then we recalculate success percentage as sum of successes of the day / sum of attempts of this day 

# In[ ]:


s['Success Percentage'] = s['Succeeded'] / s['Attempted']


# In[ ]:


s.info()


# ### Fourth : merging the 2 datasets 

# #### as stated in the introduction , we need to study the effect of weather on success percentage , so we need to merge the 2 datasets
# 
# #### since we have for some dates weather statistics only and for some other dates we have climb statistics only , so we choose outer mode for merging  

# In[ ]:


s_pooled = pd.merge(right=data_wheather  , left = s , how = 'outer', on ='Date')


# In[ ]:


s_pooled.info()


# #### regarding missing values in the pooled dataset due to merging , we assume days in which no records for climbing data , we assume they had zero attempts , successes 

# In[ ]:


s_pooled.head()


# In[ ]:


for i in ['Attempted' , 'Succeeded' , 'Success Percentage'    ] : 
    s_pooled[i] = np.where ((np.isnan(s_pooled[i]) == 1) &  (np.isnan(s_pooled['Battery Voltage AVG']) == 0) , 0 , s_pooled[i])


# In[ ]:


s_pooled.info()


# ### Fifth : studying number of attempts 

# #### we first make sure that date column is the index of our dataset 

# In[ ]:


s_pooled=s_pooled.set_index('Date')


# #### we plot the series of attempt 

# In[ ]:


Attempt = s_pooled['Attempted']
Attempt.head()


# In[ ]:


Attempt.plot(figsize=(15, 6))
plt.show()


# #### from the above plot we notice that attempts act as a seasonal time series , so we 'd better to study it using ARIMA time series process , not to study it using regression and weather columns as independent variables 

# In[ ]:


Attempt.index


# #### first we analyze the attempts series by  Seasonal decomposition using moving averages, we set freq = 30 since we deal with date data , so season here is month 

# In[ ]:


decomposition = sm.tsa.seasonal_decompose(Attempt, model='additive' , freq = 30)
decomposition.plot()

plt.show()


# #### then we use ARIMA models , for more check the following link :https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b
# 
# #### the idea is we fit several models for our series , and take the model with the least AIC , as the best fitting model for the series 

# In[ ]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x{}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[ ]:


import warnings
warnings.simplefilter(action='ignore')


# In[ ]:


a =[]
b=[]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(Attempt,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            c=param
            e=param_seasonal
            d=results.aic
            a.append((c,e))
            b.append(d)
        except:
            continue


# #### to choose the best model , we put all models and their diagnostics in a dataset , choose the least AIC

# In[ ]:


models=pd.DataFrame({'model':a,'AIC':b})


# In[ ]:


models.head()


# In[ ]:


models.loc[models['AIC'] == models.AIC.min(),:]


# #### since we got the best timeseries model , we fit it , plot the diagnostics

# In[ ]:


mod = sm.tsa.statespace.SARIMAX(Attempt,
                                order=(1, 0, 1),
                                seasonal_order=(0, 0,1, 12),
                                enforce_stationarity=False,
                                )
results = mod.fit()
print(results.summary().tables[1])


# #### we notice that all parameters are significant except seasonal moving average 

# In[ ]:


results.plot_diagnostics(figsize=(16, 8))
plt.show()


# #### We notice that model is fitting well , residuals are close to normality 

# ### Sixth : Predict Success percentage

# #### We will work on success percentage as a dependent column on weather columns , so we first do some descriptive statistics , distribution plots for each column of them 

# In[ ]:


features =['Battery Voltage AVG' , 'Temperature AVG' ,'Relative Humidity AVG' ,'Wind Speed Daily AVG' ,'Wind Direction AVG','Solare Radiation AVG'] 
for i in features:
  print('distribution of ' , i)
  print(s_pooled.loc[(np.isnan(s_pooled[i]) ==0)][i].describe())
  sns.distplot(s_pooled.loc[(np.isnan(s_pooled[i]) ==0)][i])
  plt.show()


# #### we notice that all columns close to normal distribution except the severe right skewness of 'wind speed AVG' and 'solar radiation AVG' , we may need log transformation to get rid of this skewness but it may lead to less goodness of fit in practice 
# #### we take only needed columns , weather columns and successful percentage in a new dataset 
# 

# In[ ]:


s_pooled2=s_pooled.loc[ :, ['Battery Voltage AVG' , 'Temperature AVG' ,'Relative Humidity AVG' ,'Wind Speed Daily AVG' ,'Wind Direction AVG','Solare Radiation AVG','Success Percentage' , 'Route']]


# In[ ]:


s_pooled2.head()


# In[ ]:


s_pooled2.info()


# #### we find that there are missing values in weather columns since we have days we have climb statistics for but with no data for weather , in this case we don't include these cases in our analysis.
# 
# #### We shoulf notice that the sample here will be onlt 464 cases which we divide to train and test data , so results shoulf be taken carefully , we would have more accurate results if we had a larger sample 
# 
# #### we have the categorical columns if route which we should include in our analysis , we will make binary column for each category to include in our analysis.

# In[ ]:


s_pooled2['Route'].value_counts()


# In[ ]:


s_pooled2 = pd.get_dummies(s_pooled2 , columns=['Route'])


# #### from the above frequencies table we take a binary column to include in our analysis for the most common 3 route patterns 
# 
# #### before that let's check correlation between the columns 

# In[ ]:


s_pooled2.loc[np.isnan(s_pooled2['Battery Voltage AVG']) ==0,['Battery Voltage AVG','Temperature AVG','Relative Humidity AVG','Wind Speed Daily AVG','Wind Direction AVG'
                ,'Solare Radiation AVG','Success Percentage']].corr()


# #### we notice that the most correlating column with success percentage is solar radiation , which makes sense , helping the climber with warmness and light 
# 
# #### also we notice high correlation between temprature and solar radiation , which violates one of regression assumptions which is absence of collinearity between regressors , so we can exclude it from analysis

# #### then let's have a look on data

# In[ ]:


s_pooled2.head()


# #### so regarding data we will predict success percentage through we will take cases only with valid data for weather columns , include columns for the most common 3 route patterns  , and exclude temprature AVG column to maintain absence of collinearity assumption for regression 

# In[ ]:


x_train = s_pooled2.loc[np.isnan(s_pooled2['Battery Voltage AVG']) ==0,['Battery Voltage AVG','Relative Humidity AVG','Wind Speed Daily AVG','Wind Direction AVG'
                ,'Solare Radiation AVG' , 'Route_Disappointment Cleaver' ,'Route_Gibralter Ledges' ,'Route_Ingraham Direct']]


# In[ ]:


x_train.head()


# In[ ]:


y_train = s_pooled2.loc[np.isnan(s_pooled2['Battery Voltage AVG']) ==0,'Success Percentage']


# In[ ]:


y_train.head()


# #### then we divide data into train and test datasets 

# In[ ]:


# Organize our data for training
X = x_train
Y = y_train
X, X_Val, Y, Y_Val = train_test_split(X, Y , train_size =0.7 , shuffle=True)


# #### then we predict successful percentage using simple linear regression 

# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, Y)
pred_0=pd.DataFrame(reg.predict(X_Val)).set_index(Y_Val.index)


# In[ ]:


pred_0=pd.DataFrame(reg.predict(X_Val)).set_index(Y_Val.index)


# In[ ]:


pred_0.columns=['pred']


# In[ ]:


pred_0['pred']=np.where(pred_0['pred'] <0 , 0 , pred_0['pred'])


# In[ ]:


# Print the r2 score
print(r2_score(Y_Val, np.round(pred_0['pred'],0)))
# Print the mse score
print(mean_squared_error(Y_Val,np.round(pred_0['pred'],0)))


# #### as we see , result is not satisfactory at all , we even notice that R2 is by close to zero and in some run times it gives negative values , we have to try another algoruthm for regression

# #### we can try XGB regressor as a powerful algorithm for regression and lead to many winning solution in kaggle competitions

# #### first we redivide data to train and test datasets

# In[ ]:


# Organize our data for training
X = x_train
Y = y_train
X, X_Val, Y, Y_Val = train_test_split(X, Y , train_size =0.7 , shuffle=True)


# #### to get the best classifier we use grid search , we fit the model on train set with several combination of parameters we grid on  with the default for other parameters , and the best model that gives the best diagnostics
# 
# #### for better performance , we start with grid search on learning rate and number of estimators 

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:




# A parameter grid for XGBoost
params = {'learning_rate': [ 0.01, 0.03 , 0.05 ,0.1] , 'n_estimators' :[100,  300 , 500 , 800 , 1000] , 'max_depth' : [i for i in range(8)]
}

# Initialize XGB and GridSearch
xgb = XGBRegressor(nthread=-1 , objective = 'reg:squarederror') 

grid1 = GridSearchCV(xgb, params)
grid1.fit(X, Y)


# In[ ]:



# Print the r2 score
print(r2_score(Y_Val, grid1.best_estimator_.predict(X_Val))) 
# Print the mse score
print(mean_squared_error(Y_Val, grid1.best_estimator_.predict(X_Val))) 


# In[ ]:


grid1.best_estimator_


# #### here we have the best estimator is with learning rate of 0.01 and number of estimators of 500
# 
# #### and in some run times it gives learning rate of 0.03 and number of estimators of 100
# 
# #### let's continue our grid search for the rest of important parameters 

# In[ ]:


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# In[ ]:


params = { 'learning_rate' : [0.01] , 'n_estimators':[500] , 'subsample' : [i/10.0 for i in range(11)] ,'min_child_weight' :[0.5 , 1 , 1.5 , 2] , 'colsample_bytree' : [i / 10.0 for i in range(11)]}
                

# Initialize XGB and GridSearch
xgb = XGBRegressor(nthread=-1 , objective = 'reg:squarederror') 

grid2 = GridSearchCV(xgb, params)
grid2.fit(X, Y)


# In[ ]:


# Print the r2 score
print(r2_score(Y_Val, grid2.best_estimator_.predict(X_Val))) 
# Print the mse score
print(mean_squared_error(Y_Val, grid2.best_estimator_.predict(X_Val))) 


# In[ ]:


grid2.best_estimator_


# #### we got the best estimator that gives us the the least MSE with a considerably high R2 
# #### let's get the importance score of features 

# In[ ]:


model=grid2.best_estimator_


# In[ ]:


ft = pd.Series(model.feature_importances_ , index=x_train.columns)



# In[ ]:


ft.plot(kind='bar')


# #### Here we see that  solar radiation AVG  has the highest importance score , followed by Route_Disappointment cleaver

# ### To sum up 
# #### we tried to take the most advantage of our data to fit a model to describe variety of attempts and successful percentage 
# #### , but these results should be taken with caution since the sample is small , and regarding SARIMA model we used to fit a model for Attempts , we would get a better model had we had a longer series
