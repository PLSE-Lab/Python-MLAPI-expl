#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # 1. Introduction

# In this project, I am going to explore datasets (looking at each currecy exchange rate as a funciton of time) and then create regression models using various algorithms (GradientDescent, SVM, etc) to predict future currecy exchange rate behaviors.

# # 2. EDA

# In[ ]:


df=pd.read_csv("/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv")


# In[ ]:


df.head()


# Looks like the "unnamed:0" is the same as our index column, so let's drop it first

# In[ ]:


df=df.drop('Unnamed: 0',axis=1)


# Let's look at the statistics of dataframe.

# In[ ]:


df.describe(include='all')


# One thing I noticed was that this result which is supposed to cotain the mean, std and min values for all currencies do not actually have anything. This is weird as we clearly have numbers in our original dataframe. It could be because these numbers are not float values. Let's check the datatype with an 'info' method.

# In[ ]:


df.info(verbose=True)


# Ah, we can see that all of our values in the dataframe are not floats but strings. Let's turn the number columns into numerical feature columns.

# In[ ]:


col_list=list(df)


# In[ ]:


#excluding the "0" the column, date column, as it will be converted to date-time object.
df[col_list[1:]] = df[col_list[1:]].apply(pd.to_numeric, errors='coerce')


# Then let's take care of Time series column data by converting it to date_time.

# In[ ]:


df['Time Serie']=pd.to_datetime(df['Time Serie'])


# Checking the data type one more time.

# In[ ]:


df.info(verbose=True)


# Now it looks correct :) Alright onto the real EDA!

# Let's change the structure of the dataframe such that we list all the currency types in one column. The same goes for the "value" attribute.

# In[ ]:


df_melted=df.melt(id_vars=["Time Serie"], 
        var_name="Currency type", 
        value_name="Value")


# In[ ]:


df_melted.head()


# Looking good. Let's plot the dataset then.

# In[ ]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

sns.set_style("white")
plt.figure(figsize=(20,10))
sns.lineplot(x="Time Serie", y="Value", hue="Currency type",palette='deep',data=df_melted)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# It seems like Korea-Won data's high value pushed other data to the bottom, making it hard to analyze.
# 
# Let's separate Korea-Won data from the orignal dataframe to distinguish it from the rest of the dataset.

# In[ ]:


df_melted_kor=df_melted.loc[df_melted['Currency type'] == "KOREA - WON/US$"]
df_melted_nonkor=df_melted.loc[df_melted['Currency type'] != "KOREA - WON/US$"]


# In[ ]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

sns.set_style("white")
fig, ax1 = plt.subplots(figsize=(20,10))
ax2 = ax1.twinx()

g=sns.lineplot(x="Time Serie", y="Value", hue="Currency type",ax=ax1,data=df_melted_nonkor)
g.legend(bbox_to_anchor=(-0.05, 1), loc=0, borderaxespad=0.)

#Using the dashes for korea data to stand out
g1=sns.lineplot(x="Time Serie", y="Value", color="coral",label="KOREA - WON/US$",ax=ax2,data=df_melted_kor)
g1.lines[0].set_linestyle("--")
g1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Based on this plot, it seems like Korea Won had experiened a large spike during year 2008 when there was a global economic crsis.
# 
# Let's create some regression models based on these datasets for each country

# # 3. Time series analysis

# Facebook released time series forecasting tool called Prophet in 2017. This was designed to analyze time-series on different time scales such as yearly, weekly and daily. Let's use this on our Korea data for time series analysis.

# Prophet requires you to have two columns, "ds" and "y" with the dates and values respectively.

# In[ ]:


kor_data = df_melted_kor.drop("Currency type",axis=1).rename(columns={'Time Serie': 'ds', 'Value': 'y'})


# In[ ]:


kor_data


# The data was split into 80% of training data and 20% of test data.

# In[ ]:


#splitting 80% training and 20% testing

train_kor_data, test_kor_data = np.split(kor_data, [int(.8*len(df))])


# Let's fit the train data set and create future dataframe for 4 years (4*365)

# In[ ]:


from fbprophet import Prophet

prophet_basic=Prophet()
prophet_basic.fit(train_kor_data)
train_kor_future= prophet_basic.make_future_dataframe(periods=4*365)


# Alright. Now we can call the predict method. This will assign each row in our 'future' dataframe a predicted value, which it names yhat. Additionally, it will show lower/upper bounds of uncertainty, called yhat_lower and yhat_upper. (ref:https://www.interviewqs.com/ddi_code_snippets/prophet_intro)
# 
# I will also plot the predicted train dataset.

# In[ ]:


train_kor_forecast=prophet_basic.predict(train_kor_future)
fig1 =prophet_basic.plot(train_kor_forecast)


# Let's also display the changepoints (points where real time series frequently have abrupt changes in their trajectories) to see if the model correctly recognized them.

# In[ ]:


from fbprophet.plot import add_changepoints_to_plot
fig = prophet_basic.plot(train_kor_forecast)
a = add_changepoints_to_plot(fig.gca(), prophet_basic, train_kor_forecast)


# It seems the model correctly recognized the abrupt change points.
# Let's look at the predicted train dataframe.

# In[ ]:


train_kor_forecast.head()


# Now extracting the yhat values and saving it to a series.

# In[ ]:


train_kor_yhat=train_kor_forecast['yhat']


# Let's grab the last 1,460 rows to compare against the test dataset.

# In[ ]:


train_kor_yhat_1460=train_kor_yhat[-1460:]
train_kor_yhat_1460


# Now let's take a look at our test dataset.

# In[ ]:


test_kor_data


# Alright. Let's compare the predicted yhat values with the test data to see how good our fitting was.
# But before we do this we need to ensure that the length of test dataframe and the yhat dataset (train_kor_yhat_1460) should match. Let's merge them first based on the "ds" values.

# In[ ]:


merged_kor_data=test_kor_data.merge(train_kor_forecast,left_on='ds',right_on='ds')


# In[ ]:


merged_kor_data.head()


# Ok. Let's separate them into a predicted dataset and test dataset one more time.

# In[ ]:


test_kor_data_2=merged_kor_data[['ds','y']].copy()
train_kor_yhat_2=merged_kor_data[['ds','yhat']].copy()


# Now we should not forget that the test dataframe had some NaN values. Let's how many null values there are.

# In[ ]:


test_kor_data_2.isnull().sum()


# We can see there are 45 NaN values. Let's fill these in with median values.

# In[ ]:


test_kor_data_2['y']=test_kor_data_2['y'].fillna(test_kor_data_2['y'].median())


# Ok now let's see how good our Prophet predictions were based on RMSE

# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt
mse=mean_squared_error(test_kor_data_2["y"],train_kor_yhat_2["yhat"])
rmse=sqrt(mse)

print("RMSE :", rmse)


# This RMSE value is not significnatly bad, given that the Korean Won data values were populated around ~1,000. Let's plot both preditions and actual test data together.

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.title("Korea Won currecy exchange rate predictions vs actual test")
predictions, = plt.plot(train_kor_yhat_2["ds"],train_kor_yhat_2["yhat"],label="Predictions")
test, = plt.plot(test_kor_data_2['ds'],test_kor_data_2['y'],label="Test")
plt.legend(loc='upper right')


# We can see that our predictions were not bad although they did not caputer the sudden drops around 2018-01. I am surpirsed how good facebook's Prophet is at analyzing time series with simple tricks like this!
