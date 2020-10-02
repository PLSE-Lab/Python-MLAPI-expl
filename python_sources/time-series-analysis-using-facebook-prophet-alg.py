#!/usr/bin/env python
# coding: utf-8

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


# **In this notebook, i am going to show  how to predict future stock prices using a time series forcasting model "Facebook Prophet"**

# In[ ]:


#lets read dataset
df=pd.read_csv("/kaggle/input/alphabet-inc-stockprice-history-dataset/GOOG.csv")
df


# **Lets plot dataset to observe close values agains dates**

# In[ ]:


import matplotlib.pyplot as plt
x=df["Date"]
y=df["Close"]
plt.plot(x,y)
plt.xlabel('Date')
plt.ylabel('Close Stock Values')
plt.show()


# **Now lets print some stastical variables such as mean, median,min, max , std using dataset's columns**

# In[ ]:


df.describe()


# **Now build the model. we will use "dates" and "close" prices are as features for our model**

# In[ ]:


#making dataset using those two columns
data=df[["Date","Close"]]
#lets rename columns
data=data.rename(columns={"Date":"ds","Close":"y"})
data.head(5)


# **we will not split dataset into tain and test, instead we feed all data  to fit the model and ask the model to predict future stock price for 2021.
# Lets import "prophet" model and build**

# In[ ]:


from fbprophet import Prophet
model=Prophet(daily_seasonality=True)
model.fit(data)


# # # # # Lets plot predictions.
# # # 1.We will ask model to predict future prices
# # # 2.Then it will visualize the predictions

# In[ ]:


future=model.make_future_dataframe(periods=365) #specifying number of days for future
prediction=model.predict(future)
model.plot(prediction)

#plot predictions now
plt.title("prediction of Googl's future stock price using prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show()


# # in above plot, black dots represent the traing period and blue line represents prediction confidence for future stock prices from aug-2020 to may-2021

# **> it shows that google's stock price is going to be higher than before. but did u notice huge drop from 2020-02 to 2020-05, its due to corona pandemic.**

# # Now lets plot the forecast components- trend, weekly, seasonality and daily components

# In[ ]:


model.plot_components(prediction)
plt.show()


# based on estimated trends,we can see that stock price is maximum on wednesdays

# In[ ]:





# In[ ]:




