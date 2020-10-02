#!/usr/bin/env python
# coding: utf-8

# ## **This is a sample of my learning on EDA. Please upvote if it is helpful. Suggestions are also appreciated.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from plotly import tools
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from IPython.display import HTML, Image

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Dataset
# Dataset used for analysis is publicly hosted dataset in kaggle. Please find it here [Hotel Booking Demand](https://www.kaggle.com/jessemostipak/hotel-booking-demand).Data has the factors which contributes to a booking of hotel.This data set contains booking information for a city hotel and a resort hotel, and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things.

# ### Load Dataset

# In[ ]:


hotel_df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')


# ### Interactive table in plotly for selected columns and top 10 records.

# In[ ]:


df = hotel_df.head(10)
df = df.loc[:, ['hotel','lead_time', "arrival_date_year", 'children', 'babies', 'meal', 'country']]
table = ff.create_table(df)

iplot(table, filename='pandas_table')


# In[ ]:


hotel_df.info()


# In[ ]:


df.describe().T


# ## Analyzing missing values in dataset

# In[ ]:


def summary(df):
    
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nas = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing = (df.isnull().sum() / df.shape[0]) * 100
    sk = df.skew()
    krt = df.kurt()
    
    print('Data shape:', df.shape)

    cols = ['Type', 'Total count', 'Null Values', 'Distinct Values', 'Missing Ratio', 'Skewness', 'Kurtosis']
    dtls = pd.concat([types, counts, nas, distincts, missing, sk, krt], axis=1, sort=False)
  
    dtls.columns = cols
    return dtls


# In[ ]:


details = summary(hotel_df)
table = ff.create_table(details)
iplot(table, filename='pandas_table')


# In[ ]:


from IPython.display import Image
Image('/kaggle/input/symmimg/651px-Relationship_between_mean_and_median_under_different_skewness.png')


# Skewness and Kurtosis show if the data is normally disctributed or not. If the skewness is equal to zero, the data is normally distributed, meaning it's symmetric. Negative values for the skewness indicate data that it's skewed left and it's left 'tail' is longer compare to the right one. And vice versa. If the data are multi-modal, then this may affect the sign of the skewness.
# 
# Many classical statistical tests and intervals depend on normality assumptions. Significant skewness and kurtosis indicate that data is not normal and it needs to be normalized.

# In[ ]:


hotel_df.columns


# ## 1. Distplots 
# ### Distplots are used for doing univariate analysis of continuous variables in dataset.  Following plosts will have kernel density  estimate(kde) and rug plotted on top of it.

# 1.1 Distplot of lead time

# In[ ]:


fig = ff.create_distplot([hotel_df.lead_time],['lead_time'],bin_size=10)
iplot(fig, filename='Lead Time Distplot')


# In[ ]:


hotel_df.head()


# In[ ]:


trace = go.Histogram(x=hotel_df['arrival_date_month'], marker=dict(color='rgb(0, 0, 100)'))

layout = go.Layout(
    title="Month wise count of bookings"
)

fig = go.Figure(data=go.Data([trace]), layout=layout)
iplot(fig, filename='histogram-freq-counts')


# In[ ]:


# get number of acutal guests by country
country_data = pd.DataFrame(hotel_df.loc[hotel_df["is_canceled"] == 0]["country"].value_counts())
country_data.index.name = "country"
country_data.rename(columns={"country": "Number of Guests"}, inplace=True)
total_guests = country_data["Number of Guests"].sum()
country_data["Guests in %"] = round(country_data["Number of Guests"] / total_guests * 100, 2)
table = ff.create_table(country_data.head())
iplot(table, filename='pandas_table')


# In[ ]:


# show on map
import plotly.express as px
guest_map = px.choropleth(country_data,
                    locations=country_data.index,
                    color=country_data["Guests in %"], 
                    hover_name=country_data.index, 
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title="Home country of guests")
guest_map.show()


# In[ ]:


import plotly.express as px
fig = px.scatter(hotel_df, x="arrival_date_month", y="lead_time", animation_frame="arrival_date_year", animation_group="reserved_room_type",
           size="adults", color="reserved_room_type", hover_name="reserved_room_type", facet_col="customer_type")
fig.show()


# In[ ]:




