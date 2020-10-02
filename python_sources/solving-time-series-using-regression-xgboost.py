#!/usr/bin/env python
# coding: utf-8

# # <font color='orange'>My First Kaggle Kernel and how I got first rank in competition </font>

# ![title](https://datahack-prod.s3.ap-south-1.amazonaws.com/__sized__/contest_cover/jantahack_-thumbnail-1200x1200-90.jpg)

# * From the research on all the Time Series Competitons on Kaggle ,it has been found that boosting models perform better as compared to the traditional approach using Statistical models like Holt Winters, Arima.
# * Research PDF Link: https://www.researchgate.net/publication/339362837_Learnings_from_Kaggle's_Forecasting_Competitions
# * Here, I am using the data from the Analytics Vidhya-JanataHack-IOT hackathon where we won the hackathon with Regression approach using mighty XGBoost.
# * You can check the problem statement here:https://datahack.analyticsvidhya.com/contest/janatahack-machine-learning-for-iot/ 

# * Don't forget to check the last part of the solution which is the main secret sauce :-).

# #### <font color='red'>Import libraries</font>

# In[ ]:


# import libraries
import numpy as np
import pandas as pd
import time
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

from xgboost import XGBRegressor
from xgboost import plot_importance

# Function to plot feature importance
def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)
import matplotlib.pyplot as plt


# #### <font color='red'>Loading the Data</font>

# In[ ]:


# Read train , test and submission csv in pandas dataframe
train=pd.read_csv('/kaggle/input/train.csv',parse_dates=['DateTime'])
test=pd.read_csv('/kaggle/input/test.csv',parse_dates=['DateTime'])
sub=pd.read_csv('/kaggle/input/sub.csv')


# In[ ]:


# let's check first 5 rows from train 
train.head()


# In[ ]:


# let's check last 5 rows from train 
train.tail()


# #### <font color='red'>Few observations by looking at training data</font>
# * We have 1 record for each hour and has details like vehicle count , junction number and unique id.
# * Train data is from 01Nov2015 to 30Jun2017
# 

# In[ ]:


# let's explore test data's first 5 and last 5 row
test.head().append(test.tail())


# * Here, Vehicles which is the vehicle count at a particular hour is the target Feature which we need to predict for the timeperiod (2017-07-01 - 2017-10-01) using Train data from(2015-11-01 - 2017-06-01)

# #### <font color='red'>Vehicle Trends wrt Time Period. </font>

# In[ ]:


train.loc[:,['DateTime','Vehicles']].plot(x='DateTime',y='Vehicles',title='Vehicle Trend',figsize=(16,4))


# #### <font color='red'>Few observations.</font>
# 
# * By Analysing our data we have found out that 2015 data has very low vehicle trend compared to the timeperiod 2017 which we are going to predict and also 2015 has only data for the month- 11&12 which has different trend compared to the month (7,8,9&10) for which we need to predict.So we decided to ignore 2015 data .
# * [Tip].The winner of https://www.kaggle.com/c/favorita-grocery-sales-forecasting has only used very recent data in the models, electing to drop older observations based on validation dataset performance.  
# * Selecting the right timeperiod data is very important in Time Series forecasting.
# 

# In[ ]:


# filtering data greater than or equal to 01 Jan 2016
train=train[train['DateTime']>='2016-01-01']


# #### <font color='red'>Concating train and test data for preprocessing</font>

# In[ ]:


# concat train, test data and mark where it is test , train 
train['train_or_test']='train'
test['train_or_test']='test'
df=pd.concat([train,test])


# #### <font color='red'>Creating Time Based Feature.This helps regression models to understand the trend in the data.</red>

# In[ ]:


# Below function extracts date related features from datetime
def create_date_featues(df):

    df['Year'] = pd.to_datetime(df['DateTime']).dt.year

    df['Month'] = pd.to_datetime(df['DateTime']).dt.month

    df['Day'] = pd.to_datetime(df['DateTime']).dt.day

    df['Dayofweek'] = pd.to_datetime(df['DateTime']).dt.dayofweek

    df['DayOfyear'] = pd.to_datetime(df['DateTime']).dt.dayofyear

    df['Week'] = pd.to_datetime(df['DateTime']).dt.week

    df['Quarter'] = pd.to_datetime(df['DateTime']).dt.quarter 

    df['Is_month_start'] = pd.to_datetime(df['DateTime']).dt.is_month_start

    df['Is_month_end'] = pd.to_datetime(df['DateTime']).dt.is_month_end

    df['Is_quarter_start'] = pd.to_datetime(df['DateTime']).dt.is_quarter_start

    df['Is_quarter_end'] = pd.to_datetime(df['DateTime']).dt.is_quarter_end

    df['Is_year_start'] = pd.to_datetime(df['DateTime']).dt.is_year_start

    df['Is_year_end'] = pd.to_datetime(df['DateTime']).dt.is_year_end

    df['Semester'] = np.where(df['Quarter'].isin([1,2]),1,2)

    df['Is_weekend'] = np.where(df['Dayofweek'].isin([5,6]),1,0)

    df['Is_weekday'] = np.where(df['Dayofweek'].isin([0,1,2,3,4]),1,0)
    
    df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
    
    return df


# In[ ]:


# extracting time related 
df=create_date_featues(df)


# #### <font color='red'>one hot encoding Junction</font>

# In[ ]:


for col in ['Junction']:
    df = pd.get_dummies(df, columns=[col])


# #### <font color='red'>Getting back train and test</font>

# In[ ]:


train=df.loc[df.train_or_test.isin(['train'])]
test=df.loc[df.train_or_test.isin(['test'])]
train.drop(columns={'train_or_test'},axis=1,inplace=True)
test.drop(columns={'train_or_test'},axis=1,inplace=True)


#  #### <font color='red'>Log transforming Vehicle to have normal distribution.</red>

# In[ ]:


train['Vehicles']=np.log1p(train['Vehicles'])


# #### <font color='red'>       Here comes the most important step in solving timeseries.</font>

# * Timeseries problems requires **time based validation** instead of generaly used kfold validation in regression problem. Kfold splits the data randomly and checking the model accuracy by predicting on timeperiod 2016 by using 2017 data makes no sense. 
# * Here we used time based validation for the time period (2017-01-01 to 2017-04-01) of 4 months, since the test set contains 4 months data to predict.

# In[ ]:


train1=train[train['DateTime']<'2017-03-01']#Train period from 2016-01-01 to 2017-02-31
val1=train[train['DateTime']>='2017-03-01'] #Month 3,4,5,6 as validtaion period


# #### <font color='red'>Why drop date feature when we can make use out of it.</font>

# In[ ]:


def datetounix(df):
    # Initialising unixtime list
    unixtime = []
    
    # Running a loop for converting Date to seconds
    for date in df['DateTime']:
        unixtime.append(time.mktime(date.timetuple()))
    
    # Replacing Date with unixtime list
    df['DateTime'] = unixtime
    return(df)
train1=datetounix(train1)
val1=datetounix(val1)

train=datetounix(train)
test=datetounix(test)


# In[ ]:


x_train1=train1.drop(columns={'ID','Vehicles'},axis=1)
y_train1=train1.loc[:,['Vehicles']]

x_val1=val1.drop(columns={'ID','Vehicles'},axis=1)
y_val1=val1.loc[:,['Vehicles']]


# #### <font color='red'>Validating the performance.</font>

# In[ ]:


ts = time.time()

model = XGBRegressor(
    max_depth=8,
    booster = "gbtree",
    n_estimators=100000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,
    seed=42,
    objective='reg:linear')

model.fit(
    x_train1, 
    y_train1, 
    eval_metric="rmse", 
    eval_set=[(x_train1, y_train1), (x_val1, y_val1)], 
    verbose=True, 
    early_stopping_rounds = 100)

time.time() - ts


# In[ ]:


#predicting validation data.
pred=model.predict(x_val1)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt
np.sqrt(mean_squared_error(np.expm1(y_val1), np.expm1(pred)))


# #### <font color='red'>Feature Importance</font>

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plot_features(model, (10,14))


#  <font color='red'>Here comes the secret sauce which pushed us from Rank2 to Rank 1.</font>
# * Reference:https://www.kaggle.com/xwxw2929/rossmann-sales-top1
# * This technique is also used by the Winner of Rossmann store sales prediction.
# * Here you can see the documentation from the winner of Rossman sales prediction:https://www.kaggle.com/c/rossmann-store-sales/discussion/18024       
# * This approach calculates the error in the predicted value and chooses the best  weight to mutiply with the prediction .

# In[ ]:


#checks error in prediction
res = pd.DataFrame(data = pd.concat([x_val1,y_val1],axis=1))
res['Prediction']= np.expm1(model.predict(x_val1))
res['Ratio'] = res.Prediction/np.expm1(res.Vehicles)
res['Error'] =abs(res.Ratio-1)
res['Weight'] = np.expm1(res.Vehicles)/res.Prediction
res.head()


# In[ ]:


#calculates best weight
pred1  = model.predict(x_val1)
print("weight correction")
W=[(0.990+(i/1000)) for i in range(20)]
S =[]
for w in W:
    error = sqrt(mean_squared_error(np.expm1(y_val1), np.expm1(pred1*w)))
    print('RMSE for {:.3f}:{:.6f}'.format(w,error))
    S.append(error)
Score = pd.Series(S,index=W)
Score.plot()
BS = Score[Score.values == Score.values.min()]
print ('Best weight for Score:{}'.format(BS))


# In[ ]:


pred=model.predict(x_val1)*1.009
np.sqrt(mean_squared_error(np.expm1(y_val1), np.expm1(pred)))


# * Validation accuracy RMSE dropped from **8.015 to 7.46**  by multiplying with error weight . This helped us to top the leaderboard.
# 
# * We have validated this particular weight by creating another validation for period(2016-7,8,9,10). It worked well there and also LB score increased.
# * Don't forget to upvote if you find this useful.

# ####  <font color='red'>Model using all train data (except 2015)</font>

# In[ ]:


x=train.drop(columns={'ID','Vehicles'},axis=1)
y=train.loc[:,['Vehicles']]
test=test.drop(columns={'ID','Vehicles'},axis=1)


# In[ ]:


model = XGBRegressor(
    max_depth=8,
    n_estimators=220,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,
    
    seed=42)

model.fit(x, y)


# In[ ]:


pred=model.predict(test)*1.009
sub['Vehicles']=np.expm1(pred)
sub.to_csv('finalsub.csv',index=False)


# #### <font color='red'>Other usefull kaggle kernels on this topic .</font>

# * https://www.kaggle.com/dlarionov/feature-engineering-xgboost
# * https://www.kaggle.com/abhilashawasthi/feature-engineering-lgb-model
# * https://www.kaggle.com/xwxw2929/rossmann-sales-top1
