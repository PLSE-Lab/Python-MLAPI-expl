#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotting libraries
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing, Holt
from matplotlib.dates import (
        MonthLocator,
        num2date,
        AutoDateLocator,
        AutoDateFormatter,
)
import gc # garbage collector

# stats models
import statsmodels.api as sm
from fbprophet import Prophet

# time libraries
import datetime

# warning libraries for debugging
import warnings

# deal with date in x-axis of plots
from pandas.plotting import register_matplotlib_converters

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# code to create time bar to run functions

# In[ ]:


import time, sys
from IPython.display import clear_output

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


# # Introduction
# I am still a novice when it comes to time series so I want to start simple. In this kernel we are interested in predicting meter values based on our time series data. We are going to ignore all the other features. Let's jump in...

# # Load Data
# import the training dataframe

# In[ ]:


get_ipython().run_line_magic('time', "train_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv')")
get_ipython().run_line_magic('time', "test_df = pd.read_csv('../input/ashrae-energy-prediction/test.csv')")


# Set Column Datatypes

# In[ ]:


# Saving some memory
d_types = {'building_id': np.int16,
          'meter': np.int8}

for feature in d_types:
    train_df[feature] = train_df[feature].astype(d_types[feature])
    
    
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], infer_datetime_format=True)


# Add log of meter values

# In[ ]:


train_df["log_meter_reading"]=np.log(train_df["meter_reading"]+.00001)


# # Evaluation Metric
# The evaluation metric for this competition is the root mean squared logarithmic error. Below I created a method that can calculate this value.

# In[ ]:


def rmsle(pred_series,true_series):
    sum_series = (np.log(pred_series+1) -         np.log(true_series+1))**2
    return np.sqrt(np.sum(sum_series))


# # Splitting the Training Dataset
# We need to split the original training data into a training and validation set. I decided to splits the training set into the first 9 months and the validation set into the last 3 months.

# In[ ]:


start_validation='2016-12-15'
train = train_df.loc[train_df["timestamp"]<start_validation,:]
valid = train_df.loc[train_df["timestamp"]>=start_validation,:]


# Some algorithms need the training data to be indexed by the time stamps.

# In[ ]:


# for the training data I want to reformat
# the dataframe so that the timestamp is the 
# index
print("reformat training data frame...")
def trainDF2timeDF(training_df):
    timeValue_df =  training_df.copy()
    timeValue_df = timeValue_df.set_index("timestamp")
    warnings.simplefilter("ignore")
    timeValue_df.index = pd.to_datetime(timeValue_df.index.values)
    return(timeValue_df)

timeIndexed_train = trainDF2timeDF(train)


# # Naive Approach: Just take the average values from the training data
# For this approach I use the average of the training data to predict the validation data. I will use this method as my baseline for more complicated modeling.

# In[ ]:


# create new data frame for this model
valid_avgVal_df = valid.copy()
# rename timestamp to signify the current meter reading time
valid_avgVal_df = valid_avgVal_df.rename(
    columns={"timestamp": "now", 
             "meter_reading": "cur_meter_reading",
            "log_meter_reading":"cur_log_meter_reading"})

# This model splits the data based on 
# building ID and model type
nbuildings=len(valid["building_id"].unique())
print("number of buildings: "+ str(nbuildings))
x=0
for b_id in list(valid["building_id"].unique()):
    update_progress(x / nbuildings)
    x+=1
    for meter_t in list(
        valid_avgVal_df.loc[valid_avgVal_df["building_id"]==b_id,"meter"].unique()):
        if(not ((b_id in train["building_id"]) and
           (meter_t in train.loc[train["building_id"]==b_id,"meter"].values))):
            print("missing!")
            print(b_id)
            print(meter_t)
            # if there is no meter reading for a specific
            # building ID then I'll just set the reading
            # to the average value of that meter given
            # all of the building IDs.
            valid.loc[((valid_avgVal_df["building_id"]==b_id) &
                valid_avgVal_df["meter"]==meter_t),"pred_meter_reading"] = \
                train.loc[train["meter"]==meter_t,"meter_reading"].mean()
        else:
            # calculate the average meter_reading values
            # for each meter given the building id
            valid_avgVal_df.loc[((valid_avgVal_df["building_id"]==b_id) &
                valid_avgVal_df["meter"]==meter_t),"pred_meter_reading"] = \
                train.loc[(
                (train["building_id"]==b_id) &
                (train["meter"]==meter_t)),"meter_reading"].mean()
update_progress(1)


# We can visual our predictions for each building and meter as such

# In[ ]:


b_i=1
m_t=0
train_bidX_meterY = train.loc[(
    (train["building_id"]==b_i) &
    (train["meter"]==m_t)),:].copy()
valid_bidX_meterY = valid.loc[(
    (valid["building_id"]==b_i) &
    (valid["meter"]==m_t)),:].copy()
pred_bidX_meterY = valid_avgVal_df.loc[(
    (valid_avgVal_df["building_id"]==b_i) &
    (valid_avgVal_df["meter"]==m_t)),:].copy()

plt.figure(figsize =(15,8))
plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
plt.plot(valid_bidX_meterY['meter_reading'], label = 'Validation')
plt.plot(pred_bidX_meterY['pred_meter_reading'], label = 'Simple Exponential Smoothing')
plt.legend(loc = 'best')


# Now let's evaluate our model.

# In[ ]:


print("Naive Approach - RMSLE value:")
print(rmsle(valid_avgVal_df["pred_meter_reading"],
           valid_avgVal_df["cur_meter_reading"]))


# This is a really high value but it provides a good baseline moving forward.

# Since each meter seems to have its own patter I am also interested in the RMSLE value for each meter.

# In[ ]:


avgVal_rmsle_list=[]
for meter_t in list(valid_avgVal_df["meter"].unique()):
        sub_valid_avgVal_df = valid_avgVal_df.loc[(
            valid_avgVal_df["meter"]==meter_t),:].copy()
        sub_rmsle = rmsle(sub_valid_avgVal_df["pred_meter_reading"],
           sub_valid_avgVal_df["cur_meter_reading"])
        sub_rmsle_df = pd.DataFrame({"meter":[meter_t],
                                   "rmsle":[sub_rmsle]})
        avgVal_rmsle_list.append(sub_rmsle_df)
avgVal_rmsle_df = pd.concat(avgVal_rmsle_list)
avgVal_rmsle_df  


# # Simple Exponential Smoothing
# In our baseline model, we took the average of past meter values to predict the future meter values. However, instead of weighing each past meter value equally we can assume that the most recent reading should probably way higher than readings from the distant past. **Simple Exponential Smoothing** uses weighted averages to give the largest weights to the most recent observations and the smallest weights to the oldest observations. The weights or "smoothing" parameter have been labeled as $\alpha$. So our forcast now looks like:
# $\hat{y}_{T+1|T}=\alpha y_{T} + \alpha(1-\alpha)y_{T-1} + \alpha(1-\alpha)^{2}y_{T-2}+...,$

# Let's try this method. We can use the auto optimization to automatically find an optimized $\alpha$ value for us.

# In[ ]:


valid_expSmooth = valid.copy()
# rename timestamp to signify the current meter reading time
valid_expSmooth = valid_expSmooth.rename(
    columns={"timestamp": "now", 
             "meter_reading": "cur_meter_reading",
            "log_meter_reading":"cur_log_meter_reading"})

# for the training data I want to reformat
# the dataframe so that the timestamp is the 
# index
print("reformat training data frame...")
def trainDF2timeDF(training_df):
    timeValue_df =  train.copy()
    timeValue_df = timeValue_df.set_index("timestamp")
    warnings.simplefilter("ignore")
    timeValue_df.index = pd.to_datetime(timeValue_df.index.values)
    return(timeValue_df)

timeIndexed_train = trainDF2timeDF(train)

# This model splits the data based on 
# building ID and model type
nbuildings=len(valid["building_id"].unique())
print("number of buildings: "+ str(nbuildings))
x=0
for b_id in list(valid["building_id"].unique()):
    update_progress(x / nbuildings)
    x+=1
    for meter_t in list(
        valid_expSmooth.loc[valid_expSmooth["building_id"]==b_id,"meter"].unique()):
        if(not ((b_id in train["building_id"]) and
           (meter_t in train.loc[train["building_id"]==b_id,"meter"].values))):
            print("missing!")
            print(b_id)
            print(meter_t)
            # if there is no meter reading for a specific
            # building ID then I'll just train
            # independent of the building ID
            sub_timeTrain_df = timeIndexed_train.loc[(
                timeIndexed_train["meter"]==meter_t),"meter_reading"].copy()
            numValid = len(valid_expSmooth.loc[(
                (valid_expSmooth["building_id"]==b_id) &
                (valid_expSmooth["meter"]==meter_t)),:])
            fit_simExpSmooth = SimpleExpSmoothing(sub_timeTrain_df).fit()
            # forecast the meter_readings
            valid_expSmooth.loc[(
                (valid_expSmooth["building_id"]==b_id) &
                (valid_expSmooth["meter"]==meter_t)),"pred_meter_reading"] = \
                fit_simExpSmooth.forecast(numValid).values
            # collect the alpha level used
            valid_expSmooth.loc[(
                (valid_expSmooth["building_id"]==b_id) &
                (valid_expSmooth["meter"]==meter_t)),"alpha"] = \
                fit_simExpSmooth.model.params['smoothing_level']
        else:
            # fit the model to the meter values of
            # this building type
            sub_timeTrain_df = timeIndexed_train.loc[(
                (timeIndexed_train["building_id"]==b_id) &
                (timeIndexed_train["meter"]==meter_t)),"meter_reading"].copy()
            numValid = len(valid_expSmooth.loc[(
                (valid_expSmooth["building_id"]==b_id) &
                (valid_expSmooth["meter"]==meter_t)),:])
            fit_simExpSmooth = SimpleExpSmoothing(sub_timeTrain_df).fit()
            # forecast the meter_readings
            valid_expSmooth.loc[(
                (valid_expSmooth["building_id"]==b_id) &
                (valid_expSmooth["meter"]==meter_t)),"pred_meter_reading"] = \
                fit_simExpSmooth.forecast(numValid).values
            # collect the alpha level used
            valid_expSmooth.loc[(
                (valid_expSmooth["building_id"]==b_id) &
                (valid_expSmooth["meter"]==meter_t)),"alpha"] = \
                fit_simExpSmooth.model.params['smoothing_level']
update_progress(1)


# Let's see how it looks compared to the true values in building 0 meter 0 

# In[ ]:


b_i=0
m_t=0
train_bidX_meterY = train.loc[(
    (train["building_id"]==b_i) &
    (train["meter"]==m_t)),:].copy()
valid_bidX_meterY = valid.loc[(
    (valid["building_id"]==b_i) &
    (valid["meter"]==m_t)),:].copy()
pred_bidX_meterY = valid_expSmooth.loc[(
    (valid_expSmooth["building_id"]==b_i) &
    (valid_expSmooth["meter"]==m_t)),:].copy()

plt.figure(figsize =(15,8))
plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
plt.plot(valid_bidX_meterY['meter_reading'], label = 'Validation')
plt.plot(pred_bidX_meterY['pred_meter_reading'], label = 'Simple Exponential Smoothing')
plt.legend(loc = 'best')


# Cool, now let me see some numbers...

# In[ ]:


print("Simple Exponential Smoothing - RMSLE value:")
print(rmsle(valid_expSmooth["pred_meter_reading"],
           valid_expSmooth["cur_meter_reading"]))


# In[ ]:


expSmooth_rmsle_list=[]
for meter_t in list(valid_expSmooth["meter"].unique()):
        sub_valid_expSmooth_df = valid_expSmooth.loc[(
            valid_expSmooth["meter"]==meter_t),:].copy()
        sub_rmsle = rmsle(sub_valid_expSmooth_df["pred_meter_reading"],
           sub_valid_expSmooth_df["cur_meter_reading"])
        sub_rmsle_df = pd.DataFrame({"meter":[meter_t],
                                   "rmsle":[sub_rmsle]})
        expSmooth_rmsle_list.append(sub_rmsle_df)
expSmooth_rmsle_df = pd.concat(expSmooth_rmsle_list)
expSmooth_rmsle_df  


# # Holt Model
# So the last model did a lot better since it weighed the more recent meter values higher than the past ones instead of weighing them all equally. The problem is that as we get further from our nearest point in the training set, our meter readings should not stay the same. **Holt's model** accounts for trend. For example, it accounts for the fact that the meter readings may be going up/down/same over time. Holt's model is useful if there is no seasonality in the time series.

# In[ ]:


# create new data frame for this model
valid_holt = valid.copy()
# rename timestamp to signify the current meter reading time
valid_holt = valid_holt.rename(
    columns={"timestamp": "now", 
             "meter_reading": "cur_meter_reading",
            "log_meter_reading":"cur_log_meter_reading"})


# This model splits the data based on 
# building ID and model type
nbuildings=len(valid["building_id"].unique())
print("number of buildings: "+ str(nbuildings))
x=0
for b_id in list(valid["building_id"].unique()):
    update_progress(x / nbuildings)
    x+=1
    for meter_t in list(
        valid_holt.loc[valid_holt["building_id"]==b_id,"meter"].unique()):
        if(not ((b_id in train["building_id"]) and
           (meter_t in train.loc[train["building_id"]==b_id,"meter"].values))):
            print("missing!")
            print(b_id)
            print(meter_t)
            # if there is no meter reading for a specific
            # building ID then I'll just train
            # independent of the building ID
            sub_timeTrain_df = timeIndexed_train.loc[(
                timeIndexed_train["meter"]==meter_t),"meter_reading"].copy()
            numValid = len(valid_holt.loc[(
                (valid_holt["building_id"]==b_id) &
                (valid_holt["meter"]==meter_t)),:])
            fit_holt = Holt(
                sub_timeTrain_df).fit(optimized=True)
            # forecast the meter_readings
            valid_holt.loc[(
                (valid_holt["building_id"]==b_id) &
                (valid_holt["meter"]==meter_t)),"pred_meter_reading"] = \
                fit_holt.forecast(numValid).values
            # collect the alpha level used
            valid_holt.loc[(
                (valid_holt["building_id"]==b_id) &
                (valid_holt["meter"]==meter_t)),"alpha"] = \
                fit_holt.model.params['smoothing_level']
        else:
            # fit the model to the meter values of
            # this building type
            sub_timeTrain_df = timeIndexed_train.loc[(
                (timeIndexed_train["building_id"]==b_id) &
                (timeIndexed_train["meter"]==meter_t)),"meter_reading"].copy()
            numValid = len(valid_holt.loc[(
                (valid_holt["building_id"]==b_id) &
                (valid_holt["meter"]==meter_t)),:])
            fit_holt = Holt(
                sub_timeTrain_df).fit(optimized=True)
            # forecast the meter_readings
            valid_holt.loc[(
                (valid_holt["building_id"]==b_id) &
                (valid_holt["meter"]==meter_t)),"pred_meter_reading"] = \
                fit_holt.forecast(numValid).values
            # collect the alpha level used
            valid_holt.loc[(
                (valid_holt["building_id"]==b_id) &
                (valid_holt["meter"]==meter_t)),"alpha"] = \
                fit_holt.model.params['smoothing_level']
update_progress(1)


# Let's see how it looks in each type of meter. I randomly chose building ID's with each type of meter

# In[ ]:


b_i=0
m_t=0
train_bidX_meterY = train.loc[(
    (train["building_id"]==b_i) &
    (train["meter"]==m_t)),:].copy()
valid_bidX_meterY = valid.loc[(
    (valid["building_id"]==b_i) &
    (valid["meter"]==m_t)),:].copy()
pred_bidX_meterY = valid_holt.loc[(
    (valid_holt["building_id"]==b_i) &
    (valid_holt["meter"]==m_t)),:].copy()

plt.figure(figsize =(15,8))
plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
plt.plot(valid_bidX_meterY['meter_reading'], label = 'Validation')
plt.plot(pred_bidX_meterY['pred_meter_reading'], label = 'Holt Model')
plt.legend(loc = 'best')


# In[ ]:


b_i=161
m_t=1
train_bidX_meterY = train.loc[(
    (train["building_id"]==b_i) &
    (train["meter"]==m_t)),:].copy()
valid_bidX_meterY = valid.loc[(
    (valid["building_id"]==b_i) &
    (valid["meter"]==m_t)),:].copy()
pred_bidX_meterY = valid_holt.loc[(
    (valid_holt["building_id"]==b_i) &
    (valid_holt["meter"]==m_t)),:].copy()

plt.figure(figsize =(15,8))
plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
plt.plot(valid_bidX_meterY['meter_reading'], label = 'Validation')
plt.plot(pred_bidX_meterY['pred_meter_reading'], label = 'Holt Model')
plt.legend(loc = 'best')


# In[ ]:


b_i=745
m_t=2
train_bidX_meterY = train.loc[(
    (train["building_id"]==b_i) &
    (train["meter"]==m_t)),:].copy()
valid_bidX_meterY = valid.loc[(
    (valid["building_id"]==b_i) &
    (valid["meter"]==m_t)),:].copy()
pred_bidX_meterY = valid_holt.loc[(
    (valid_holt["building_id"]==b_i) &
    (valid_holt["meter"]==m_t)),:].copy()

plt.figure(figsize =(15,8))
plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
plt.plot(valid_bidX_meterY['meter_reading'], label = 'Validation')
plt.plot(pred_bidX_meterY['pred_meter_reading'], label = 'Holt Model')
plt.legend(loc = 'best')


# In[ ]:


b_i=106
m_t=3
train_bidX_meterY = train.loc[(
    (train["building_id"]==b_i) &
    (train["meter"]==m_t)),:].copy()
valid_bidX_meterY = valid.loc[(
    (valid["building_id"]==b_i) &
    (valid["meter"]==m_t)),:].copy()
pred_bidX_meterY = valid_holt.loc[(
    (valid_holt["building_id"]==b_i) &
    (valid_holt["meter"]==m_t)),:].copy()

plt.figure(figsize =(15,8))
plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
plt.plot(valid_bidX_meterY['meter_reading'], label = 'Validation')
plt.plot(pred_bidX_meterY['pred_meter_reading'], label = 'Holt Model')
plt.legend(loc = 'best')


# Number time!

# In[ ]:


print("Holt - RMSLE value:")
print(rmsle(valid_holt["pred_meter_reading"],
           valid_holt["cur_meter_reading"]))


# In[ ]:


holt_rmsle_list=[]
for meter_t in list(valid_holt["meter"].unique()):
        sub_valid_holt_df = valid_holt.loc[(
            valid_holt["meter"]==meter_t),:].copy()
        sub_rmsle = rmsle(sub_valid_holt_df["pred_meter_reading"],
           sub_valid_holt_df["cur_meter_reading"])
        sub_rmsle_df = pd.DataFrame({"meter":[meter_t],
                                   "rmsle":[sub_rmsle]})
        holt_rmsle_list.append(sub_rmsle_df)
holt_rmsle_list = pd.concat(holt_rmsle_list)
holt_rmsle_list  


# # Winter Holt Model
#  **Winter Holt's model**  takes both trend and seasonality into account. Seasonality is found when time series data show variations a specific regular intervals less than a year (eg. hourly, weekly, monthly, quarterly). Unfortunately, Quarterly may be difficult to see with only a years worth of data, but keep an eyes out for other patterns in your EDA.  

# Here, since we split our training and validation set into 9 months in the training set and 3 months in the validation set. 

# In[ ]:


# Winter-Holt's prediction model
# parameters
# - train_dataframe: dataframe containing training data
# - timeIdx_train: data frame with time series in the index
# - valid_winterHolt: copy of the validation data frame
# - seasonility: a list of all the seasonal period that are being tested
# - b_id: building ID
# - meter_t: meter number
# - pred_col: list of the predicion column names (must be unique)
# - plot: set to True to print a plot of the predictions
def winHolt(train_dataframe, timeIdx_train, valid_winterHolt, 
            seasonality, b_id, meter_t, plot=False, 
            pred_col=[], known_true=False):
    if len(seasonality) > len(pred_col):
        if len(seasonality) ==1:
            pred_col=["winterHolt"]
        else:
            pred_col=[]
            for i in range(0,len(seasonality)):
                pred_col.append("winterHolt_sp_"+str(seasonality[i]))
    ignored_pred_cols=[]
    if(not ((b_id in train_dataframe["building_id"]) and
           (meter_t in train_dataframe.loc[train_dataframe["building_id"]==b_id,"meter"].values))):
        print("missing!")
        print(b_id)
        print(meter_t)
        # if there is no meter reading for a specific
        # building ID then I'll just train
        # independent of the building ID
        sub_timeTrain_df = timeIdx_train.loc[(
            timeIdx_train["meter"]==meter_t),"meter_reading"].copy()
        numValid = len(valid_winterHolt.loc[(
            (valid_winterHolt["building_id"]==b_id) &
            (valid_winterHolt["meter"]==meter_t)),:])
        for i in range(0,len(seasonality)):
            if (len(sub_timeTrain_df.index.unique())/
                 seasonality[i]) >= 2:
                fit_wintHolt = ExponentialSmoothing(
                    sub_timeTrain_df,
                    seasonal_periods=seasonality[i],
                    trend='add',
                    seasonal='add').fit()
                # forecast the meter_readings
                valid_winterHolt.loc[(
                    (valid_winterHolt["building_id"]==b_id) &
                    (valid_winterHolt["meter"]==meter_t)),pred_col[i]] = \
                    fit_wintHolt.forecast(numValid).values
            else:
                ignored_pred_cols.append(pred_col[i])
    else:
        # fit the model to the meter values of
        # this building type
        sub_timeTrain_df = timeIdx_train.loc[(
                        (timeIdx_train["building_id"]==b_id) &
                        (timeIdx_train["meter"]==meter_t)),"meter_reading"].copy()
        numValid = len(valid_winterHolt.loc[(
                        (valid_winterHolt["building_id"]==b_id) &
                        (valid_winterHolt["meter"]==meter_t)),:])
        for i in range(0,len(seasonality)):
            if (len(sub_timeTrain_df.index.unique())/
                 seasonality[i]) >= 2:
                fit_wintHolt = ExponentialSmoothing(
                    sub_timeTrain_df,
                    seasonal_periods=seasonality[i],
                    trend='add',
                    seasonal='add').fit()
                # forecast the meter_readings
                valid_winterHolt.loc[(
                    (valid_winterHolt["building_id"]==b_id) &
                    (valid_winterHolt["meter"]==meter_t)),pred_col[i]] = \
                    fit_wintHolt.forecast(numValid).values
            else:
                ignored_pred_cols.append(pred_col[i])
    if plot:
        b_i=b_id
        m_t=meter_t
        train_bidX_meterY = train_dataframe.loc[(
            (train_dataframe["building_id"]==b_i) &
            (train_dataframe["meter"]==m_t)),:].copy()
        
        valid_bidX_meterY = valid_winterHolt.loc[(
            (valid_winterHolt["building_id"]==b_i) &
            (valid_winterHolt["meter"]==m_t)),:].copy()
        plt.figure(figsize =(15,8))
        plt.plot(train_bidX_meterY['meter_reading'], label = 'Train')
        if known_true:
            plt.plot(valid_bidX_meterY['cur_meter_reading'], label = 'Validation')
        for i in range(0,len(seasonality)):
            if pred_col[i] not in ignored_pred_cols:
                plt.plot(valid_bidX_meterY[pred_col[i]], label = pred_col[i])
        plt.legend(loc = 'best')
    return(valid_winterHolt)    


# In[ ]:


# create new data frame for this model
valid_winterHolt = valid.copy()
# rename timestamp to signify the current meter reading time
valid_winterHolt = valid_winterHolt.rename(
    columns={"timestamp": "now", 
             "meter_reading": "cur_meter_reading",
            "log_meter_reading":"cur_log_meter_reading"})


# In[ ]:


# set seasonality
sp=[365-90,4*9,9,3]


# This model splits the data based on 
# building ID and model type
nbuildings=len(valid["building_id"].unique())
print("number of buildings: "+ str(nbuildings))
x=0
for b_id in list(valid["building_id"].unique()):
    
    update_progress(x / nbuildings)
    x+=1
    for meter_t in list(
        valid_winterHolt.loc[valid_winterHolt["building_id"]==b_id,"meter"].unique()):
        print(b_id)
        print(meter_t)
        valid_winterHolt = winHolt(train,
            timeIndexed_train, valid_winterHolt,
            sp, b_id,meter_t)
update_progress(1)


# Let's see how it looks compared to the true values in meters of some of the buildings

# In[ ]:


sp=[365-90,4*9,9,3]
b_id=0
meter_t=0
valid_winterHolt = winHolt(train, timeIndexed_train, valid_winterHolt, sp, b_id,meter_t, True, known_true=True)


# In[ ]:


sp=[365-90,4*9,9,3]
b_id=161
meter_t=1
valid_winterHolt = winHolt(train, timeIndexed_train, valid_winterHolt, sp, b_id,meter_t, True, known_true=True)


# In[ ]:


sp=[365-90,4*9,9,3]
b_id=745
meter_t=2
valid_winterHolt = winHolt(train, timeIndexed_train, valid_winterHolt, sp, b_id,meter_t, True, known_true=True)


# In[ ]:


sp=[365-90,4*9,9,3]
b_id=106
meter_t=3
valid_winterHolt = winHolt(train, timeIndexed_train, valid_winterHolt, sp, b_id,meter_t, True, known_true=True)


# Number time!

# In[ ]:


pred_col=[]
for i in sp:
    pred_colName = "winterHolt_sp_"+str(i)
    print("winterHolt (sp ="+str(i)+") - RMSLE value:")
    print(rmsle(valid_winterHolt[pred_colName],
           valid_winterHolt["cur_meter_reading"]))
    pred_col.append(pred_colName)


# In[ ]:


winterHolt_rmsle_list=[]
for i in range(0,len(sp)):
    for meter_t in list(valid_winterHolt["meter"].unique()):
        sub_valid_winterHolt_df = valid_winterHolt.loc[(
            valid_winterHolt["meter"]==meter_t),:].copy()
        sub_rmsle = rmsle(sub_valid_winterHolt_df[pred_col[i]],
           sub_valid_winterHolt_df["cur_meter_reading"])
        sub_rmsle_df = pd.DataFrame({"seasonality":[sp[i]],
                                     "meter":[meter_t],
                                   "rmsle":[sub_rmsle]})
        winterHolt_rmsle_list.append(sub_rmsle_df)
winterHolt_rmsle_list = pd.concat(winterHolt_rmsle_list)
print(winterHolt_rmsle_list)  


# # Summary

# To summarize our findings for each meter:
# * we predicted meter0 (electricity) with the smallest error rate (rmsle~1557) using Holt's model which ignores seasonality
# * we predicted meter1 (chilledwater) best using Winter-Holt's model and weekly smoothing (seasonality=36, rmsle~2045) or seasonal smoothing (seasonality=3, rmsle~2049)
# * meter2 (steam) works best by Winter Holt's model and daily smoothing (seasonality=275, rmsle~1390)
# * meter3 (hotwater) works best by Winter Holt's model and seasonal smoothing (seasonality=3, rmsle~1117)

# # Work I Learned From
# * https://www.youtube.com/watch?v=d4Sn6ny_5LI (Time Series Forecasting Video)
