#!/usr/bin/env python
# coding: utf-8

# Hello this kernel is a work in process.I will be doing an exploratory data analysis and Crime forecasting in this data set.I will be updating this kernel in coming days.If you like my work please do vote.

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


# #### Importing Python Modules 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
import folium 
from folium.plugins import HeatMap
from fbprophet import Prophet


# In[ ]:


data=pd.read_csv('../input/crimes-in-boston/crime.csv',encoding='latin-1')


# In[ ]:


data.head()


# In[ ]:


data.shape


# #### Renaiming the columns for convinence

# In[ ]:


data1 = data.rename(columns={'OFFENSE_CODE':'Code','OFFENSE_CODE_GROUP':'Group','OFFENSE_DESCRIPTION':'Description','OCCURRED_ON_DATE':'Date'})
data1.head()


# #### Finding the missing data 

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(data1.isnull(),cbar=False,cmap='YlGnBu')
plt.ioff()


# All the horizontal lines show the missing data in the dataset.Let us clean up the missing data by dropping the columns with missing data.Also the columns getting dropped will not have an affect on the forecasting of the crime.

# #### Dropping the unwanted columns

# In[ ]:


data1.drop(['INCIDENT_NUMBER','Code','SHOOTING','UCR_PART','Lat','Long','Location'],inplace=True,axis=1)
data1.head()


# #### Arranging the date in datetime format

# In[ ]:


data1['Date']=pd.to_datetime(data1['Date'])
data1.head()


# In[ ]:


data1.Date


# In[ ]:


data1.index=pd.DatetimeIndex(data1.Date)
data1.head()


# #### Getting information most recurring Crime

# In[ ]:


data1['Group'].value_counts()


# In[ ]:


data1['Group'].value_counts().iloc[:15]


# In[ ]:


order_data=data1['Group'].value_counts().iloc[:15].index
plt.figure(figsize=(15,10))
sns.countplot(y='Group',data=data1,order=order_data)
plt.ioff()


# So we can clearly see that the motor vehicle accident,theft and medical assistance are most crime indidents reported in Boston.

# #### Resampling the data: 
# This is done to segregate the crime count based on time period like month,quarter and year

# In[ ]:


data1.resample('Y').size()


# In[ ]:


plt.plot(data1.resample('Y').size())
plt.title('Crime Count Per Year')
plt.xlabel('Years')
plt.ylabel('Number of Crimes')
plt.ioff()


# In[ ]:


plt.plot(data1.resample('M').size())
plt.title('Crime Count Per Month')
plt.xlabel('Months')
plt.ylabel('Number of Crimes')
plt.ioff()


# In[ ]:


plt.plot(data1.resample('Q').size())
plt.title('Crime Count Per Quarter')
plt.xlabel('Quaterly')
plt.ylabel('Number of Crimes')
plt.ioff()


# The X axis values and not getting displayed correctly.I am not sure why ? May be someone can throw some light on what needs to be done

# #### Preparing the data 

# In[ ]:


Boston_prophet=data1.resample('M').size().reset_index()


# In[ ]:


Boston_prophet.head()


# #### Renaming the columns of Boston_prophet

# In[ ]:


Boston_prophet.columns=['Date','Crime_Count']


# In[ ]:


Boston_prophet.head()


# In[ ]:


Boston_prophet_final=Boston_prophet.rename(columns={'Date':'ds','Crime_Count':'y'})


# In[ ]:


Boston_prophet_final.head()


# #### Make Predictions

# In[ ]:


m=Prophet()
m.fit(Boston_prophet_final)


# In[ ]:


future=m.make_future_dataframe(periods=365)
forecast=m.predict(future)


# In[ ]:


forecast


# In[ ]:


#figure=m.plot(forecast,xlabel='Data',ylabel='Crime Rate')


# In[ ]:


#figure=m.plot_components(forecast)


# In[ ]:




