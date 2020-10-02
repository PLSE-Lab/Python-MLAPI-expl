#!/usr/bin/env python
# coding: utf-8

# Some years back I went for a trip to Kodaikanal in Tamil Nadu.There we had a Fruit Milk Shake which the locals refered to as Butter Fruit.The fruit indeed tasted like Butter.The memory of the fruit remained fresh in my Memory.After many years I found out this fruit are called as Avacado.In this kernel we will be exploring and vizualizaing the dataset and try to Forecast the price of Avacados.This kernel is a work in process.If you like it please do vote.

# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
img=np.array(Image.open('../input/butter-fruit/Butter_fruit.jpg'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.show()


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


# In[ ]:


df=pd.read_csv('../input/avocado-prices/avocado.csv')
df.head()


# In the data set we can see Region the Average price of Avacado every week.Price is that of a single Avacado.

# **Summary of the DataSet**

# In[ ]:


print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nUnique values :  \n',df.nunique())


# Luckly there are no missing values in out datset 

# In[ ]:


df.info()


# In[ ]:


df.describe().T


# Average Price of One avacado is 1.40 Dollars minimum price and maximum price being 0.44 and 3.25 respectively.

# **Which Columns have Catogerical Data?**

# In[ ]:


df.select_dtypes(exclude=['int','float']).columns


# **Importing the Python Modules**

# In[ ]:


import random 
import seaborn as sns
from fbprophet import Prophet


# **Exploring the dataset**

# In[ ]:


df.tail()


# We need to sort the data properly for further analysis

# In[ ]:


df=df.sort_values('Date')
df.tail()


# We have data from Jan 2015 to Mar 2018

# **Average Price**

# In[ ]:


df.plot(x='Date', y='AveragePrice',legend=True,figsize=(20,4));
plt.ioff()


# **Different Regions**

# In[ ]:


plt.figure(figsize=[25,12])
sns.countplot(x='region',data=df);
plt.xticks(rotation=45)
plt.ioff()


# **Count based on year**

# In[ ]:


sns.countplot(x='year',data=df);


# We see that that the bar for the year 2018 is short as we have data only upto march for the year 2018

# * **Forecasting the Price of Avacado **

# In[ ]:


df_prophet=df[['Date','AveragePrice']]
df_prophet


# We have extracted the the Date and Average Price as this two information are only needed for us to forecast the price of Avacado

# **Renaiming Columns to suite Prophet Algorithm**

# In[ ]:


df_prophet=df_prophet.rename(columns={'Date':'ds','AveragePrice':'y'})
df_prophet


# In[ ]:


m=Prophet()
m.fit(df_prophet)


# In[ ]:


future=m.make_future_dataframe(periods=365)
forecast=m.predict(future)
forecast


# **Plotting the forecast**

# In[ ]:


figure=m.plot(forecast,xlabel='Date',ylabel='Price')


# So we can see from the graph that we had price for Avacado from Jan-2015 to Mar-2018.With the Prophet algorithm we are able to forecast the prices for the next years.We can see that the prediction is that the prices will go down.

# **Plotting the components of the Forecast**

# In[ ]:


figure=m.plot_components(forecast)


# 1.The first graph shows the overall trend of the price in the data set and for the future forecast for one year.
# 
# 2.The second graph show the seasonality of the price in a year,We can see that the price of Avacados Peak in the month of October

# **Lets do a region wise Forecast**

# In[ ]:


df.columns


# In[ ]:


df['region'].unique()


# Lets select California as our region

# In[ ]:


region_sample=df[df['region']=='California']
region_sample.head()


# Now we have only Data for California in our Dataset.

# In[ ]:


region_sample=region_sample.sort_values('Date')


# **Average Price in California**

# In[ ]:


region_sample.plot(x='Date', y='AveragePrice',legend=True,figsize=(20,4));
plt.ioff()


# * **Forecasting the Price of Avacado for Region of California **

# In[ ]:


region_sample=region_sample[['Date','AveragePrice']]
region_sample


# **Renaiming the Columns**

# In[ ]:


region_sample=region_sample.rename(columns={'Date':'ds','AveragePrice':'y'})
region_sample


# In[ ]:


m=Prophet()
m.fit(region_sample)
future=m.make_future_dataframe(periods=365)
forecast=m.predict(future)
forecast


# **Plotting the Forcast**

# In[ ]:


figure=m.plot(forecast,xlabel='Date',ylabel='Price')


# So from the forecast Plot we can see that the Price of Avacado would increase in the coming year that is after Mar 2018.

# **Plotting the Components of the Forecast**

# In[ ]:


figure=m.plot_components(forecast)


# The trend also shows that there will be increase in price of Avacado in California
# 
# The seasonality curve follows almost the same trend in California as in the rest of the country
