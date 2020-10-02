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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Let us load the data and index using date.
# > We use forward filling to fill the missing and invalid values.

# In[ ]:


data = pd.read_csv('../input/AirQualityUCI_req.csv')
data.index = pd.DatetimeIndex(data.Date, dayfirst=True).strftime('%Y-%m-%d')
data = data.drop(['Date' ], 1)
cols = data.columns
data = data[data[cols] > 0]
data = data.fillna(method='ffill')
data.head()


# We now group the data by date and take the mean of the values. 
# 
# *This gives us average values of each feature over a day.*

# In[ ]:


daily_data = data.groupby(data.index).mean()
daily_data.head()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sb
sb.set(style="darkgrid")


# Let us start with a Dendogram. Dendogram is used to reveal hierarchies in the data and features. 
# > Looking at the vertical Dendograms helps us to find the correlation between the features.
# 1.  - In the dendogam obtained, the features *PT08.S5(O3), PT08.S1(CO) and PT08.S2(NMHC)* are closely related. ( Due to their close spacing in the Dendogram. 
#     - Similarly, *NMHC(GT) and NOx(GT)* are related to each other. 
#     - Also, *T, C6H6 and CO* are closely related to each other.

# In[ ]:


sb.clustermap(daily_data)


# > Let us choose features accordingly and plot the Dendogram again.

# In[ ]:


cols = ['NO2(GT)', 'C6H6(GT)', 'PT08.S4(NO2)', 'PT08.S3(NOx)', 'PT08.S5(O3)']
sb.clustermap(daily_data[cols])


# > Now, let us emphasize the relationships between these variables. 
# - We specifically compare the features with NO2(GT).

# In[ ]:


g = sb.jointplot("C6H6(GT)", "NO2(GT)", data = daily_data, kind="reg")


# In[ ]:


g = sb.jointplot("PT08.S4(NO2)", "NO2(GT)", data = daily_data, kind="reg")


# In[ ]:


g = sb.jointplot("PT08.S3(NOx)", "NO2(GT)", data = daily_data, kind="reg")


# In[ ]:


g = sb.jointplot("PT08.S5(O3)", "NO2(GT)", data = daily_data, kind="reg")


# > From the above plots, we can infer that there is no clear linear relationship between the features. 
# 
# *Futher, we can eliminate PT08.S5(O3) and PT08.S3(NOx)*

# Let us plot scatter plots of these features to see the distribution w.r.t eacch other

# In[ ]:


cols = ['NO2(GT)', 'C6H6(GT)', 'PT08.S4(NO2)']
sb.pairplot(daily_data[cols])


# > Let us visualise how the selected features behave over time.

# In[ ]:


set1 = ['NO2(GT)']
set2 = ['C6H6(GT)' ]
set3 = ['PT08.S4(NO2)']


# In[ ]:


sb.lineplot(data=daily_data[set1], linewidth=2.5)


# In[ ]:


sb.lineplot(data=daily_data[set2], linewidth=2.5)


# In[ ]:


sb.lineplot(data=daily_data[set3], linewidth=2.5)


# Let us now visualise how the data varies based on the day of the week and month of the year.

# In[ ]:


import datetime
daily_data['Date'] = daily_data.index
daily_data['Day'] = daily_data['Date'].map( lambda x : datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%A'))
daily_data['Month'] = daily_data['Date'].map( lambda x : datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%m'))
daily_data.head()


# Aggreagating based on the day and finding the mean. We reindex to get the data in the order of occurence of days naturally.

# In[ ]:


cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_data = daily_data.groupby('Day').mean()
weekly_data['Day'] = weekly_data.index
weekly_data = weekly_data.reindex(cats)
weekly_data


# The following plots describe the trends based on week of the day.

# In[ ]:


sb.barplot(x="Day", y=set1[0], data=weekly_data)


# In[ ]:


sb.barplot(x="Day", y=set2[0], data=weekly_data)


# In[ ]:


sb.barplot(x="Day", y=set3[0], data=weekly_data)


# In[ ]:


monthly_data = daily_data.groupby('Month').mean()
monthly_data['Month'] = monthly_data.index
monthly_data


# The following plots describe the trends of the features across months of the year.

# In[ ]:


sb.barplot(x="Month", y=set1[0], data=monthly_data)


# In[ ]:


sb.barplot(x="Month", y=set2[0], data=monthly_data)


# In[ ]:


sb.barplot(x="Month", y=set3[0], data=monthly_data)


# We use lineplots to visualize the trends of the features.

# In[ ]:


d = weekly_data[set1[0]].values
plt.plot(d)
plt.ylabel(set1[0])
plt.xticks([i for i in range(len(d))], cats, rotation=20)


# In[ ]:


d = weekly_data[set2[0]].values
plt.plot(d)
plt.ylabel(set2[0])
plt.xticks([i for i in range(len(d))], cats, rotation=20)


# In[ ]:


d = weekly_data[set3[0]].values
plt.plot(d)
plt.ylabel(set3[0])
plt.xticks([i for i in range(len(d))], cats, rotation=20)


# Similarly, we plot monthly trend for the features

# In[ ]:


months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# In[ ]:


d = monthly_data[set1[0]].values
plt.plot(d)
plt.ylabel(set1[0])
plt.xticks([i for i in range(len(d))], months, rotation=20)


# In[ ]:


d = monthly_data[set2[0]].values
plt.plot(d)
plt.ylabel(set2[0])
plt.xticks([i for i in range(len(d))], months, rotation=20)


# In[ ]:


d = monthly_data[set3[0]].values
plt.plot(d)
plt.ylabel(set3[0])
plt.xticks([i for i in range(len(d))], months, rotation=20)


# - From the above plots, we can infer that Sunday is the least polluted day and Friday is the most polluted day.
# - Interestingly, the effect of pollutants is spread across months. There is exists high concentration of a pollutant even in the absence of other pollutants in all the months.
