#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb;
import matplotlib.pyplot as plt

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sklearn.metrics import mean_squared_error
from math import sqrt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Index

# 1. About this DataSet
# 2. Introduction
# 3. EDA
# 4. Time series analysis

# # 1. About this DataSet

# This datasets were generated on the Federal Reserve's Download Data Program. Some changes were made in the dataset, such as header simplifications and inversions of base currency. For example, Fed provides Australian Dollar, Euro, New Zeland Dollar and United Kingdom Pound based in their units (not in dollar). So I made a convertion for this dataset in order to view all rates based in dollar units.

# # 2. Introduction

# In this project, I am going to looking at each currecy exchange rate as a funciton of time and then create regression models using various algorithms to predict future currecy exchange rate behaviors.

# # 3.EDA
# 

# In[ ]:


Cy_File=pd.read_csv("/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv")


# In[ ]:


Cy_File.head(10)


# Let'S Clean our data:
# * Since Unnamed:0 is not required we can remove it from our data.

# In[ ]:


Cy_File=Cy_File.drop("Unnamed: 0",axis=1)
Cy_File.head(2)


# describe() is used to view some basic statistical details like percentile, mean, std etc. of a data frame or a series of numeric values. When this method is applied to a series of string, it returns a different output 

# In[ ]:


Cy_File.describe()


# we have numbers in our original dataframe. So the result should be containing mean, std and min values for all currencies but are missing from the result.It could be because these numbers are not float values.

# Let's check the datatype with an 'info' method.

# In[ ]:


Cy_File.info()


# Our values in the dataframe should be in floats but are strings. Let's turn the number columns into numerical feature columns.

# In[ ]:


List=list(Cy_File)


# In[ ]:


Cy_File[List[1:]]=Cy_File[List[1:]].apply(pd.to_numeric ,errors='coerce')
#Coverting the elements into numeric format
Cy_File['Time Serie']=pd.to_datetime(Cy_File['Time Serie'])
#Then let's take care of Time series column data by converting it to date_time.
Cy_File.info()


# In[ ]:


#Creating a new Data Set
dataSet=Cy_File.melt(id_vars=["Time Serie"],var_name="Currency type",value_name="Value")
dataSet.head()


# In[ ]:


sb.set()
sb.set_style("white")
plt.figure(figsize=(20,10))
sb.lineplot(x="Time Serie", y="Value", hue="Currency type",palette='deep',data=dataSet)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)


# It seems like Korea-Won data's making it hard to analyze. Let's separate Korea-Won data from the orignal dataframe.
# 
# 

# In[ ]:


data_Set_Kor=dataSet.loc[dataSet['Currency type'] == "KOREA - WON/US$"]
data_Set_rest=dataSet.loc[dataSet['Currency type'] != "KOREA - WON/US$"]


# In[ ]:


sb.set_style("white")
fig, ax1 = plt.subplots(figsize=(20,10))
ax2 = ax1.twinx()

#Using the dashes for not included korea data to stand out
Plot=sb.lineplot(x="Time Serie", y="Value", hue="Currency type",ax=ax1,data=data_Set_rest)
Plot.legend(bbox_to_anchor=(-0.05, 1), loc=0, borderaxespad=0.)

#Using the dashes for korea data to stand out
Plot2=sb.lineplot(x="Time Serie", y="Value", color="coral",label="KOREA - WON/US$",ax=ax2,data=data_Set_Kor)
Plot2.lines[0].set_linestyle("--")
Plot2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# As we can see on this plot, it seems like Korea Won had experiened a large spike during year 2008 when there was a global economic crsis.
# 

# # 4. Time series analysis

# We will be using time series forecasting tool called "Prophet". Using this we can analyze on differnt time scale Yearly,Monthly,Weekly. Lets use this tool on Korean Data for Time series analysis.

# In[ ]:


data_Korea = data_Set_Kor.drop("Currency type",axis=1).rename(columns={'Time Serie': 'ds', 'Value': 'y'})
#Prophet requires you to have two columns, "ds" and "y" with the dates and values respectively.


# In[ ]:


data_Kor_Train,data_Kor_Test=np.split(data_Korea,[int(.8*len(Cy_File))])
#Let's Split data into 80% of training data and 20% of test data.


# Let's fit the train data set and create future dataframe for 5 years (5*365)
# 

# In[ ]:


prophet_basic=Prophet()
prophet_basic.fit(data_Kor_Train)
future_train_kor= prophet_basic.make_future_dataframe(periods=5*365)


# Lets plot the predicted train dataset.

# In[ ]:


forecast_train_kor=prophet_basic.predict(future_train_kor)
Plot =prophet_basic.plot(forecast_train_kor)


# Lets show points where real time series frequently have abrupt changes in their trajectories to check whether the model work's corectly or not

# In[ ]:


fig = prophet_basic.plot(forecast_train_kor)
a = add_changepoints_to_plot(fig.gca(), prophet_basic, forecast_train_kor)


# It look's like the model correctly recognized the abrupt change points.

# In[ ]:


forecast_train_kor.head()
#Let's look at the predicted train dataframe.


# In[ ]:


#Now extracting the yhat values and saving it to a series.
train_kor_yhat=forecast_train_kor['yhat']
train_kor_yhat


# In[ ]:


#Look at our test dataset.
data_Kor_Test


# Comparing the predicted yhat values with the test data to see how good our fitting was.

# In[ ]:


merged_data_korea=data_Kor_Test.merge(forecast_train_kor,left_on='ds',right_on='ds')


# Ok. Let's separate them into a predicted dataset and test dataset one more time.
# 

# In[ ]:


korea_data_test2=merged_data_korea[['ds','y']].copy()
kor_yhat_train2=merged_data_korea[['ds','yhat']].copy()


# * Some of values still have "NaN" as their values we need to replace them with the mean of the column.
# * Let's how many null values there are.

# In[ ]:


korea_data_test2.isnull().sum()


# In[ ]:


korea_data_test2['y']=korea_data_test2['y'].fillna(korea_data_test2['y'].median())


# Ok now let's see how good our Prophet predictions were based on RMSE
# 

# In[ ]:


mse=mean_squared_error(korea_data_test2["y"],kor_yhat_train2["yhat"])
rmse=sqrt(mse)

print("RMSE :", rmse)


# This RMSE value is not significnatly bad, given that the Korean Won data values were populated around ~1,000. 

# In[ ]:


#Let's plot both preditions and actual test data together.
plt.figure(figsize=(25,10))
plt.title("(Korea currecy exchange rate) predictions vs actual test")
predictions, = plt.plot(kor_yhat_train2["ds"],kor_yhat_train2["yhat"],label="Predictions")
test, = plt.plot(korea_data_test2['ds'],korea_data_test2['y'],label="Actual Test")
plt.legend(loc='lower right')


# We can see that our predictions were not bad although they did not predict the sudden drops around 2018-01.
