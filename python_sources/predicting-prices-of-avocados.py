#!/usr/bin/env python
# coding: utf-8

# **Hi, in this notebook I would like to look at the avocado data and by using fbprophet library predict future prices. **

# In[ ]:


import numpy as np 
import pandas as pd 
from fbprophet import Prophet

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/avocado.csv")


# After loading our data let's inspect it a little bit. 

# In[ ]:


df.head()


# As we can see we have a column describing type of a avocado, what are these types?

# In[ ]:


df.groupby('type').groups


# We got two types of avocados - conventional and organic, I have chosen the conventional ones to make predictions for. 

# In[ ]:


PREDICTION_TYPE = 'conventional'
df = df[df.type == PREDICTION_TYPE]


# After we have loaded our libraries and data, we shall continue by converting Date column of our DataFrame to pandas readable type.  

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])


# Let's check how many distinct regions column region includes and how many entries each region has.

# In[ ]:


regions = df.groupby(df.region)
print("Total regions :", len(regions))
print("-------------")
for name, group in regions:
    print(name, " : ", len(group))


# Each one of the regions has exactly 169 entries. The next step is to choose a region, for which we would like to predict future prices.

# In[ ]:


PREDICTING_FOR = "TotalUS"


# In[ ]:


date_price = regions.get_group(PREDICTING_FOR)[['Date', 'AveragePrice']].reset_index(drop=True)


# In[ ]:


date_price.plot(x='Date', y='AveragePrice', kind="line")


# Rename dataframe for fbprophet lib.

# In[ ]:


date_price = date_price.rename(columns={'Date':'ds', 'AveragePrice':'y'})


# Creating & fitting a model. All of the code here is further explained on prophet[ quick start page](https://facebook.github.io/prophet/docs/quick_start.html).

# In[ ]:


m = Prophet()
m.fit(date_price)


# In[ ]:


future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[ ]:


forecast.tail()


# In[ ]:


fig1 = m.plot(forecast)


# Here we can see how components of the model affect our predictions. 

# In[ ]:


fig2 = m.plot_components(forecast)


# Thanks for reading my notebook.
