#!/usr/bin/env python
# coding: utf-8

# This is the 1st out of 3 notebooks where I have plotted the **meter_reading over the year 2016 (duration of training data) for different types of meters for all the buildings across sites 11 to 15**. The goal is to quickly refer to the meter_reading for all the sites to identify outliers and gaps.
# 
# - [Plot for Site 0 to 5](https://www.kaggle.com/arnabbiswas1/eda-meter-reading-for-site-0-5)
# - [Plot for Site 6 to 10](https://www.kaggle.com/arnabbiswas1/eda-meter-reading-for-site-6-10)
# 
# Note: I have used plotly for this EDA so that the exact value of meter reading at a particular time stamp can be quickly referred from the figure. For the buildings which just have only one type of meter, to display the meter type, please hover the mouse over the line plot. For buildings with multiple meter types, different types of meters are explicitly mentioned along with the color code.
# 
# **Unfortunately, because of the size, this notebook takes few minutes to get rendered completely. So, please wait for sometime once you open the page.**

# In[ ]:


import numpy as np 
import pandas as pd
import plotly
import plotly.graph_objects as go


# In[ ]:


train_df = pd.read_pickle('../input/ashrae-data-minification/train.pkl')
building_df = pd.read_pickle('../input/ashrae-data-minification/building_metadata.pkl')
train_weather_df = pd.read_pickle('../input/ashrae-data-minification/weather_train.pkl')


# In[ ]:


temp_df = train_df[['building_id']]
temp_df = temp_df.merge(building_df, on=['building_id'], how='left')
del temp_df['building_id']
train_df = pd.concat([train_df, temp_df], axis=1)

del building_df, temp_df

temp_df = train_df[['site_id','timestamp']]
temp_df = temp_df.merge(train_weather_df, on=['site_id','timestamp'], how='left')
del temp_df['site_id'], temp_df['timestamp']
train_df = pd.concat([train_df, temp_df], axis=1)

del train_weather_df, temp_df


# In[ ]:


meter_dict = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}

def plot_meter_reading_for_site(df, site_id):
    building_id_list = df[df.site_id == site_id].building_id.unique().tolist()
    for building_id in building_id_list:
        fig = go.Figure()
        df_subset = df[(df.site_id == site_id) & (df.building_id == building_id)]
        meter_list = df_subset.meter.unique().tolist()
        for meter in meter_list:
            df_super_subset = df_subset[df_subset.meter == meter]
            fig.add_trace(go.Scatter(
                 x=df_super_subset.timestamp,
                 y=df_super_subset.meter_reading,
                 name=f"{meter_dict[meter]}",
                 hoverinfo=f'x+y+name',
                 opacity=0.7))

            fig.update_layout(width=700,
                            height=500,
                            title_text=f"Meter Reading for Site [{site_id}] Building [{building_id}]",
                            xaxis_title="timestamp",
                            yaxis_title="meter_reading",)
        fig.show()


# # meter_reading site 11

# In[ ]:


plot_meter_reading_for_site(train_df, 11)


# # meter_reading site 12

# In[ ]:


plot_meter_reading_for_site(train_df, 12)


# # meter_reading site 13

# In[ ]:


plot_meter_reading_for_site(train_df, 13)


# # meter_reading site 14

# In[ ]:


plot_meter_reading_for_site(train_df, 14)


# # meter_reading site 15

# In[ ]:


plot_meter_reading_for_site(train_df, 15)

