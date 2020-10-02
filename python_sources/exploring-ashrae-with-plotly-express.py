#!/usr/bin/env python
# coding: utf-8

#  ## Welcome 
# 
# In this notebook we are going to explore ASHRAE dataset with library plotly-express. Please feel free to clone or comment about any mistakes I have done.

# In this competetion we are required to predict the energy consumption of different buildings given in the dataset.

# ![ASHRAE](data:image/png;base64,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)

# In[ ]:


# import the require libraries
import pandas as pd
import numpy as np
import os

import plotly_express as px
from plotly.offline import init_notebook_mode,iplot
from plotly import graph_objects as go
init_notebook_mode(connected=True)


# Let's investigate the dataset!
# 
# There are 5 files namely train,test,building metadata, weather train, weather test. I believe train and test the files over we run the alogorithms.
# 
# But lets check how building metadata and weather can help

# ### Building Meta-Data

# In[ ]:


df_meta = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")
df_meta.head()


# In[ ]:


df_meta.describe()


# In[ ]:


df_meta.info()


# We can see that there a lot of null values in floor count, so lets not take into much of conisderation

# In[ ]:


tr1 = px.histogram(df_meta ,x = 'year_built',
                   marginal="violin",
                   title='Histogram of Number of building built in an year',
                   labels={'year_built':'Year'}, # can specify one label per df column
                   opacity=0.8,
#                     width=5000
                  )
tr1.show()


# In[ ]:


df_meta_pri = df_meta.groupby('primary_use',as_index=False)['building_id'].count().rename(
    columns={'building_id':'n_buildings'}).sort_values('n_buildings',ascending=False)
tr2 = px.bar(
    df_meta_pri,
    title = 'No of bulidngs for each type',
    x = 'primary_use',
    y = 'n_buildings',
)
tr2.show()


# In[ ]:


tr3 = px.histogram(df_meta ,x = 'square_feet',
                   marginal="violin",
                   title='Histogram of sq ft area of buildings',
                   labels={'square_feet':'Area in sq ft'}, # can specify one label per df column
                   opacity=0.8,
#                     width=5000
                  )
tr3.show()


# In[ ]:


tr4 = px.scatter(df_meta, y="square_feet",
                 x="year_built", color="primary_use",
                 size="square_feet",
                 title = ""
                )
tr4.show()


# More to come. Please wait

# ### Weather Data

# In[ ]:


df_weather = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')
df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
df_weather['site_id'] = df_weather['site_id'].astype('str')

df_weather.head()


# In[ ]:


tr5 = px.line(df_weather,
           x = "timestamp",
           y = "air_temperature",
           color = 'site_id',
#                  line_shape="spline"
       
          )
tr5.show()


# As we can see there is definetly a seasonality in the weather features as expected. Now we need to find out how this effects on the electricity usage!

# In[ ]:


df_train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")
# df_test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

df_train.head()


# In[ ]:


from fbprophet import Prophet


# #### Simple prediction with fbprophet
# 
# lets select one of meter type in one of the building and try to predict using fbprophet without major parameter tuning

# In[ ]:


df_train_10 = df_train[df_train['building_id'] == 10].reset_index(drop=True).rename(columns={'timestamp':'ds','meter_reading':'y'})

df_train_10['ds'] = pd.to_datetime(df_train_10['ds'])
print(f"No of unique meter types :{df_train_10['meter'].nunique()}")
df_train_10.head(2)


# As there is only one meter type, we need not filter for a specific meter type again!
# 
# Let's create a sample test-train data set from the train dataset.

# In[ ]:


n = 200
df_10_train = df_train_10.iloc[:-n]
df_10_test = df_train_10.iloc[-n:]

m = Prophet(yearly_seasonality=False,)
m.fit(df_10_train)
future = df_10_test[['ds']]

results = m.predict(future)


# In[ ]:


fig = m.plot_components(results)


# In[ ]:


results_new = pd.merge(results,df_10_test,on='ds',how='left')


# In[ ]:


tr6 = go.Scatter(x=results_new.ds,
                y = results_new.yhat.values,name='preds')
tr7 = go.Scatter(x=results_new.ds,
                y = results_new.y.values,name='actuals')
tr8 = go.Scatter(x=results_new.ds,
                y = df_weather[df_weather['site_id'] == '0'].iloc[-n:]['air_temperature'].values)
print("Actuals vs Preds")
iplot([tr6,tr7])
print("Distribution of air temperature for the same time period")
iplot([tr8])


# From the above charts we can see that there is a strong daily seasonality, And looks like there could be a relation between few metrics of weather to a respective meter type. Lets investigate on it as next step.
# 
# 

# In[ ]:


del df_train


# In[ ]:


df_weather


# In[ ]:


df_weather_1 = df_weather[df_weather['site_id']=="1"]


# In[ ]:


corr = df_weather_1.drop('precip_depth_1_hr',axis=1).corr()
corr.style.background_gradient(cmap='coolwarm')


# In[ ]:




