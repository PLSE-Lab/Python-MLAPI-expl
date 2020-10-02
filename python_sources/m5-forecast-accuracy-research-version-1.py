#!/usr/bin/env python
# coding: utf-8

# # M5 Forecast Accuracy Research
# 
# The M5 Forecasting competition provides a real life data set to perform business analysis and forecast research. My goal, as a Stern freshman student, is to learn the various time series forecasting algorithms, and compare their accuracy in different types of use cases, similar to the M4 research published here. https://mofc.unic.ac.cy/m4/.
# 
# Many thanks to Professor Makridakis and the MOFC team for making the data easily available and pulling the talents from all over the world to explore and develop the wonderful world of telling the future!
# 
# During the next month, I plan to compare the following commonly used forecasting algorithms, using the Root Mean Squared Scaled Error (RMSSE) as the measurement:
# 
# * Autoregressive Integrated Moving Average (ARIMA)
# * Seasonal ARIMA (SARIMA)
# * Seasonal ARIMA with Excogenous Regressors (SARIMAX)
# * Simple Exponential Smoothing (SES)
# * Holt Winter's Exponential Smoothing (Holt)
# 
# The algorithms are explained in this Jason Brownlee here https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
# 
# As my goal is not to compete, but to learn and compare the algorithms, I plan to analyze only one store (TX_1) and aggregate at the product category level (Food, Hobbies and Household). I will skip the price dataset but will see how the events helps the forecast accuracy. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Load the data and take a glance at the data

# In[ ]:


# Step 1: get the data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

CalendarDF=pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv", header=0)
PricesDF=pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv", header=0)
SalesDF=pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv", header=0)
SubmissionDF=pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv", header=0)


# In[ ]:


CalendarDF.info()


# In[ ]:


PricesDF.info()


# In[ ]:


SalesDF.info()


# In[ ]:


SubmissionDF.info()


# # I don't plan to use the price data. Deleting the PriceDF and SubmissionDF.

# In[ ]:


import os, psutil  

pid = os.getpid()
py = psutil.Process(pid)
memory_use = py.memory_info()[0] / 2. ** 30
print ('memory GB:' + str(np.round(memory_use, 2)))


# In[ ]:


del PricesDF
del SubmissionDF

import gc
gc.collect()


# # Prepare my dataset for TX_1 and aggregate to the product category level

# In[ ]:


TX_1_Sales = SalesDF[['TX_1' in x for x in SalesDF['store_id'].values]]
TX_1_Sales = TX_1_Sales.reset_index(drop = True)
TX_1_Sales.info()


# In[ ]:


# Generate MultiIndex for easier aggregration.
TX_1_Indexed = pd.DataFrame(TX_1_Sales.groupby(by = ['cat_id','dept_id','item_id']).sum())
TX_1_Indexed.info()


# In[ ]:


# Aggregate total sales per day for each sales category
Food = pd.DataFrame(TX_1_Indexed.xs('FOODS').sum(axis = 0))
Hobbies = pd.DataFrame(TX_1_Indexed.xs('HOBBIES').sum(axis = 0))
Household = pd.DataFrame(TX_1_Indexed.xs('HOUSEHOLD').sum(axis = 0))
Food.info()


# In[ ]:


## Merge the aggregated sales data to the calendar dataframe based on date
CalendarDF = CalendarDF.merge(Food, how = 'left', left_on = 'd', right_on = Food.index)
CalendarDF = CalendarDF.rename(columns = {0:'Food'})
CalendarDF = CalendarDF.merge(Hobbies, how = 'left', left_on = 'd', right_on = Hobbies.index)
CalendarDF = CalendarDF.rename(columns = {0:'Hobbies'})
CalendarDF = CalendarDF.merge(Household, how = 'left', left_on = 'd', right_on = Household.index)
CalendarDF = CalendarDF.rename(columns = {0:'Household'})
CalendarDF.head(10)


# In[ ]:


# Store a copy of the new data frame for my future use. Delete the SalesDF to save memory usage.
CalendarDF.to_csv("CalendarDF.csv", index = False)
del SalesDF
gc.collect()
memory_use = py.memory_info()[0] / 2. ** 30
print ('memory GB:' + str(np.round(memory_use, 2)))


# # Do some graphing to look at the data.
# I referenced this notebook for the graphing. https://www.kaggle.com/risheshg/m5-accuracy-starter-data-exploration. There are a lot more to learn in plotting the data. But for now, I will pause.
# 
# For next week, I plan to start use the CalendarDF to do time-series analysis. 

# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('bmh')

CalendarDF['date'] = pd.to_datetime(CalendarDF.date)
## graph daily sales data for each year 
is_2011 = CalendarDF.year == 2011
is_2012 = CalendarDF.year == 2012
is_2013 = CalendarDF.year == 2013
is_2014 = CalendarDF.year == 2014
is_2015 = CalendarDF.year == 2015
is_2016 = CalendarDF.year == 2016

temp = CalendarDF[is_2011]
temp2 = CalendarDF[is_2012]
temp3 = CalendarDF[is_2013]
temp4 = CalendarDF[is_2014]
temp5 = CalendarDF[is_2015]
temp6 = CalendarDF[is_2016]

fig, axs = plt.subplots(3 , 2, figsize = (14,10))
axs[0,0].title.set_text('2011')
axs[0 , 1].title.set_text('2012')
axs[1 , 0].title.set_text('2013')
axs[1 , 1].title.set_text('2014')
axs[2 , 0].title.set_text('2015')
axs[2 , 1].title.set_text('2016')

axs[0,0].plot(temp.date, temp.Food)
axs[0,0].plot(temp.date, temp.Hobbies)
axs[0,0].plot(temp.date, temp.Household)

df = temp[temp.event_name_1.notnull()]
df.reset_index(drop=True)

l1 = list(df['date'])
l2 = list(df['event_name_1'])

start, end = axs[0,0].get_ylim()[0], axs[0,0].get_ylim()[1]

axs[0,0].vlines(l1, start, end, linestyles = '--', color = 'r', alpha = .5)
for i in range(0,len(l2)):
    if l2[i] is not None:
       axs[0,0].text(l1[i],end,l2[i])
        
axs[0,1].plot(temp2.date, temp2.Food)
axs[0,1].plot(temp2.date, temp2.Hobbies)
axs[0,1].plot(temp2.date, temp2.Household)

axs[1,0].plot(temp3.date, temp3.Food)
axs[1,0].plot(temp3.date, temp3.Hobbies)
axs[1,0].plot(temp3.date, temp3.Household)

axs[1,1].plot(temp4.date, temp4.Food)
axs[1,1].plot(temp4.date, temp4.Hobbies)
axs[1,1].plot(temp4.date, temp4.Household)

axs[2,0].plot(temp5.date, temp5.Food)
axs[2,0].plot(temp5.date, temp5.Hobbies)
axs[2,0].plot(temp5.date, temp5.Household)

axs[2,1].plot(temp6.date, temp6.Food)
axs[2,1].plot(temp6.date, temp6.Hobbies)
axs[2,1].plot(temp6.date, temp6.Household)

plt.show()

