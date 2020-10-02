#!/usr/bin/env python
# coding: utf-8

# **This kernel mainly covers:**<br>
# <font color='#0073e6'>
#     <ol>
#         <b><li>Data Cleaning</li></b>
#         <b><li>Data Munging</li></b>
#         <b><li>EDA</li></b>
#     </ol>
# </font>

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


path = '/kaggle/input/temperature-readings-iot-devices/'
temp_iot_data = pd.read_csv(path + 'IOT-temp.csv')


# In[ ]:


temp_iot_data.shape


# In[ ]:


temp_iot_data.info()


# In[ ]:


temp_iot_data.head()


# In[ ]:


temp_iot_data.rename(columns = {'room_id/id':'room_id', 'out/in':'out_in'}, inplace = True)


# In[ ]:


temp_iot_data.head()


# In[ ]:


def get_df_summary(df):
    
    '''This function is used to summarise especially unique value count and data type for variable'''
    
    unq_val_cnt_df = pd.DataFrame(df.nunique(), columns = ['unq_val_cnt'])
    unq_val_cnt_df.reset_index(inplace = True)
    unq_val_cnt_df.rename(columns = {'index':'variable'}, inplace = True)
    unq_val_cnt_df = unq_val_cnt_df.merge(df.dtypes.reset_index().rename(columns = {'index':'variable', 0:'dtype'}),
                                          on = 'variable')
    unq_val_cnt_df = unq_val_cnt_df.sort_values(by = 'unq_val_cnt', ascending = False)
    
    return unq_val_cnt_df


# In[ ]:


unq_val_cnt_df = get_df_summary(temp_iot_data)


# In[ ]:


unq_val_cnt_df


# Let us drop **room_id** variable as it contains one unique value only and hence won't be of any use in our data analysis.

# In[ ]:


temp_iot_data.drop(columns = 'room_id', inplace = True)


# In[ ]:


print('No. of duplicate records in the data set : {}'.format(temp_iot_data.duplicated().sum()))


# In[ ]:


# Check for duplicate records.

temp_iot_data[temp_iot_data.duplicated()]


# In[ ]:


temp_iot_data.loc[temp_iot_data['id'] == '__export__.temp_log_196108_4a983c7e']


# As per the requirement, **id** variable is supposed to contain unique values for each reading, rows identified above with same id value can easily be pronounced as **duplicate rows**.

# In[ ]:


# Drop duplicate records.

temp_iot_data = temp_iot_data.drop_duplicates()


# **Variable : noted_date**

# Most important point to note here is, **noted_date** variable has date-time values without **seconds** though its clearly mentioned in the requirement that the data has been recorded at seconds level.

# In[ ]:


# Convert noted_date into date-time.

temp_iot_data['noted_date'] = pd.to_datetime(temp_iot_data['noted_date'], format = '%d-%m-%Y %H:%M')


# In the absence of **seconds** component from **noted_date** variable values, the given data set would give a perception of **Data Duplicacy or Data Redundancy** for the combination of **noted_date, out_in & temp** variables. <br>
# 
# **How do we handle this situation?**<br> 
# Find out a variable (if any) in the given data set that can be used to sort the data such that the sorted data set is in the order as if it is sorted by noted_date variable. 

# In[ ]:


# Check data duplicacy based on noted_date variable.

temp_iot_data.groupby(['noted_date'])['noted_date'].count().sort_values(ascending = False).head()


# In[ ]:


temp_iot_data.loc[temp_iot_data['noted_date'] == pd.to_datetime('2018-09-12 03:09:00', format = '%Y-%m-%d %H:%M:%S'), ].sort_values(by = 'id').head(10)


# If we take a closer look at the values of **id** variable, it has **numeric** and **alpha-numeric** values embedded in it which we may use as a primary key. We need to further analyse to determine if the numeric part can be used as a sort order to sort the given data set. Definitely, the alpha-numeric part will not help us in sorting the data.<br><br>
# For e.g. "__export__.temp_log_196134_bd201015". It has two very distinct values: **196134** and **bd201015** which we need to analyse further to re-confirm if any one of these can be used as unique identifier.<br>

# **Consider the numerical part** and see if it can be used to uniquely identify the observations and sort the data.

# In[ ]:


# Check if last but one bit of "id" can be used as primary key.

temp_iot_data['id'].apply(lambda x : x.split('_')[6]).nunique() == temp_iot_data.shape[0]


# Yes, the numerical part of **id** can be used as a primary key to uniquely identify the observations in the given data set.

# Let's further analyse numerical part of **id** in order to re-confirm if it can be used to sort the given data set.

# In[ ]:


# Create a new column to store last but one bit of id value.

temp_iot_data['id_num'] = temp_iot_data['id'].apply(lambda x : int(x.split('_')[6]))


# Closer look at values of id_num variable for a specific noted_date is going to give us some more insights on how data is recorded.

# In[ ]:


temp_iot_data.loc[temp_iot_data['noted_date'] == pd.to_datetime('2018-09-12 03:09:00', format = '%Y-%m-%d %H:%M:%S'), ].sort_values(by = 'id_num').head(10)


# After sorting the data on **id_num** variable, we see gaps in values between 17003 to 17006 and 17006 to 17009.

# In[ ]:


temp_iot_data.loc[temp_iot_data['id_num'].isin(range(17003, 17007))].sort_values(by = 'id_num')


# Some of the important observations we can make here are:
# 
# 1. There is no data for id_num = 17005. 
# 2. Also, id_num = 17004 observation has been recorded at "2018-09-12 03:08:00" while observation for id_num = 17003 has been recorded at "2018-09-12 03:09:00". However, it's expected to have noted_date for former later to id_num = 17003.

# In[ ]:


temp_iot_data.loc[temp_iot_data['id_num'].isin(range(17006, 17010))].sort_values(by = 'id_num')


# We do not have data for id_nums 17007 through 17008. Missing id_num values is really not a road block in sorting the data.

# In[ ]:


temp_iot_data.loc[temp_iot_data['noted_date'] == pd.to_datetime('2018-09-09 16:24:00', format = '%Y-%m-%d %H:%M:%S'), ].sort_values(by = 'id_num').head(10)


# In[ ]:


temp_iot_data.loc[temp_iot_data['id_num'].isin(range(4000, 4003))].sort_values(by = 'id_num')


# As seen before, we see a gap in id_num values.

# In[ ]:


temp_iot_data.loc[temp_iot_data['id_num'].isin(range(4002, 4007))].sort_values(by = 'id_num')


# Observation with id_num = 4004 is expected to have noted_date same as that of 4002 and 4006.

# Based on the above observations, we can conclude that **id_num** variable cannot be used to sort the data set to get the actual data in sorted order in spite of absence of seconds component in the **noted_date** variable.<br><br>
# We'll use **id_num** as primary key to identify the observations uniquely. Replace **id** variable values with **id_num** and drop **id_num** variable from the data set.

# In[ ]:


temp_iot_data.loc[:, 'id'] = temp_iot_data.loc[:, 'id_num']


# In[ ]:


# Drop id_num column from the data set.

temp_iot_data.drop(columns = 'id_num', inplace = True)


# In[ ]:





# In[ ]:


print('No. of years data : {}'.format(temp_iot_data['noted_date'].dt.year.nunique()))


# In[ ]:


print('No. of months data : {}'.format(temp_iot_data['noted_date'].dt.month.nunique()))


# In[ ]:


sorted(temp_iot_data['noted_date'].dt.month.unique())


# We have got data for only second half of the year 2018.

# In[ ]:


print('No. of days data : {}'.format(temp_iot_data['noted_date'].dt.day.nunique()))


# **Variable : month**

# In[ ]:


temp_iot_data['month'] = temp_iot_data['noted_date'].apply(lambda x : int(x.month))


# In[ ]:


# temp_iot_data['month'].unique()


# **Variable : day**

# In[ ]:


temp_iot_data['day'] = temp_iot_data['noted_date'].apply(lambda x : int(x.day))


# In[ ]:


# print(sorted(temp_iot_data['day'].unique()))


# **Variable : day_name**

# In[ ]:


temp_iot_data['day_name'] = temp_iot_data['noted_date'].apply(lambda x : x.day_name())


# In[ ]:


# print(temp_iot_data['day_name'].unique())


# **Variable : hour**

# In[ ]:


temp_iot_data['hour'] = temp_iot_data['noted_date'].apply(lambda x : int(x.hour))


# In[ ]:


print(sorted(temp_iot_data['hour'].unique()))


# In[ ]:


temp_iot_data.head()


# Let's assume this data has been recorded in India. Based on this assumption, we can presume two very important things:<br><br>
# **(A) Climatological Seasons:**<br>
# &emsp;&emsp;India Meteorological Department (IMD) follows the international standard of four climatological seasons with some local adjustments:<br>
# 
# &emsp;&emsp;a. Winter (December, January and February).<br>
# &emsp;&emsp;b. Summer (March, April and May).<br>
# &emsp;&emsp;c. Monsoon means rainy season (June to September).<br>
# &emsp;&emsp;d. Post-monsoon period (October to November).<br>
# 
# Accordingly, we will create another variable **season** to hold the season which we are going derive based on **month** variable value.<br><br>
# 
# **(B) Unit of measurement used to measure **temp**.
# 
# &emsp;&emsp;As India follows SI units system of measurement, we assume that the temperature is recorded in degree celsius.

# **Variable : season**

# In[ ]:


def map_month_to_seasons(month_val):
    if month_val in [12, 1, 2]:
        season_val = 'Winter'
    elif month_val in [3, 4, 5]:
        season_val = 'Summer'
    elif month_val in [6, 7, 8, 9]:
        season_val = 'Monsoon'
    elif month_val in [10, 11]:
        season_val = 'Post_Monsoon'
    
    return season_val


# In[ ]:


temp_iot_data['season'] = temp_iot_data['month'].apply(lambda x : map_month_to_seasons(x))


# In[ ]:


temp_iot_data['season'].value_counts(dropna = False)


# Since, we have data for 2nd half of year 2018 only, we see Monsoon, Post_Monsoon and Winter in season variable. 

# In[ ]:


temp_iot_data.head()


# **Variable : month_name**

# In[ ]:


temp_iot_data['month_name'] = temp_iot_data['noted_date'].apply(lambda x : x.month_name())


# In[ ]:


# temp_iot_data['month_name'].value_counts(dropna = False)


# **Variable : Timing**

# Let's bin the **hour** into four different timings i.e. **Night**, **Morning**, **Afternoon** and **Evening**.<br>
# 
# - Night : 2200 - 2300 Hours & 0000 - 0359 Hours
# - Morning : 0400 - 1159 Hours
# - Afternoon : 1200 - 1659 Hours
# - Evening : 1700 - 2159 Hours

# In[ ]:


def bin_hours_into_timing(hour_val):
    
    if hour_val in [22,23,0,1,2,3]:
        timing_val = 'Night (2200-0359 Hours)'
    elif hour_val in range(4, 12):
        timing_val = 'Morning (0400-1159 Hours)'
    elif hour_val in range(12, 17):
        timing_val = 'Afternoon (1200-1659 Hours)'
    elif hour_val in range(17, 22):
        timing_val = 'Evening (1700-2159 Hours)'
    else:
        timing_val = 'X'
        
    return timing_val


# In[ ]:


temp_iot_data['timing'] = temp_iot_data['hour'].apply(lambda x : bin_hours_into_timing(x))


# In[ ]:


temp_iot_data['timing'].value_counts(dropna = False)


# In[ ]:


del unq_val_cnt_df


# In[ ]:





# **<font color='#cc6699'>Very important to tell yourself all the time :** 
# 
# ### <font color = '#0099ff'> Do the Data Analysis for Inside and Outside temperatures, separately. 

# **How is overall temperature variation across months inside and outside room?**

# In[ ]:


fig = px.box(temp_iot_data, x = 'out_in', y = 'temp', labels = {'out_in':'Outside/Inside', 'temp':'Temperature'})
fig.update_xaxes(title_text = 'In or Out')
fig.update_yaxes(title_text = 'Temperature (in degree celsius)')
fig.update_layout(title = 'Overall Temp. Variation Inside-Outside Room')
fig.show()


# **Observations:** (assuming there are no outliers)<br>
# - Temperature recorded inside room : <br>
# &nbsp;Min. temperature : **21&deg;C**.<br>
# &nbsp;Max. temperature : **41&deg;C**.<br>
# - Temperature recorded outside room : <br>
# &nbsp;Min. temperature : **24&deg;C**.<br>
# &nbsp;Max. temperature : **51&deg;C**.<br>
# - Average tempurature recorded inside the room < Average temperature recorded outside the room. This is an obvious thing to observe.
# - Temperature has varied alot outside room when compared to inside.
# - Outside room : Magnitude of temperature variation before and after **37&deg;C** is almost same. However, temperature has varied a lot after reaching **40&deg;C** in comparison to temperature variation upto **31&deg;C**.

# **How temperature varies across seasons?**

# In[ ]:


fig = px.box(temp_iot_data, 
             x = 'season', 
             y = 'temp', 
             color = 'out_in', 
             labels = {'out_in':'Outside/Inside', 'temp':'Temperature', 'season':'Season'})
fig.update_xaxes(title_text = 'Inside/Outside - Season')
fig.update_yaxes(title_text = 'Temperature (in degree celsius)')
fig.update_layout(title = 'Season-wise Temp. Variation')
fig.show()


# **Observations:** (assuming there are no outliers)<br>
# 1. Max. temperature of 51&deg;C has been recorded in Monsoon season which is quite surprising and not expected in rainy season.<br>
# <i>Note: We have to yet see when was this temperature was recorded; Is it at the start of monsoon season or at the end of the season?</i>
# 2. As usual the lowest temperature of 21&deg;C has been recorded in Winter season.<br>
# 3. Magnitude of temperature variation is observed **inside room** in **Monsoon season is higher** compared to Winter and Post Monsoon season.<br>
# 4. Similary, maximum temperature variation **outside room** is **observed in Monsoon season**.
# 5. In comparison to average (median) temperatures of inside and outside room of Winter and Post-Monsoon seasons, average (median) temperature recorded **inside room** is higher to that of **outside room** in the same season.

# **How temperature varies across month?**

# In[ ]:


fig = px.box(temp_iot_data, x = 'month_name', y = 'temp', 
             category_orders = {'month_name':['July', 'August', 'September', 'October', 'November', 'December']},
             color = 'out_in')
fig.update_xaxes(title_text = 'Inside/Outside Month')
fig.update_yaxes(title_text = 'Temperature (in degree celsius)')
fig.update_layout(title = 'Monthly Temp. Variation')
fig.show()


# In[ ]:


round(temp_iot_data['month_name'].value_counts(dropna = False) * 100 / temp_iot_data.shape[0],1)


# **Observations:** (assuming there are no outliers)<br>
# 1. Volume of data we have for **July** and **August** months is **very very low** compared to other months.
# 2. **Maximum temperature variations** are observed in **September** month **both inside and outside the room**.
# 3. **Highest average temperature** (median) of **39&deg;C** is observed in **November** months.
# 4. **Lowest temperature** of **21&deg;C** is recorded in **December** month.
# 5. Despite of Point No. 1, **Minimum temperature variation** is observed in **July** and **August** months.

# **How temperature varies for different timings for all seasons?**

# In[ ]:


temp_iot_data.head()


# In[ ]:


for in_out_val in ['In', 'Out']:

    fig = px.box(temp_iot_data.loc[temp_iot_data['out_in'] == in_out_val], x = 'month_name', y = 'temp', 
                 category_orders = {'month_name':['July', 'August', 'September', 'October', 'November', 'December'], 
                                    'timing':['Morning (0400-1159 Hours)', 'Afternoon (1200-1659 Hours)', 'Evening (1700-2159 Hours)', 'Night (2200-0359 Hours)']},
                 hover_data = ['hour'],
                 labels = {'timing':'Timing', 'hour':'Hour', 'month_name':'Month', 'temp':'Temperature'},
                 color = 'timing')
    fig.update_xaxes(title_text = 'Month-Day Timings')
    fig.update_yaxes(title_text = 'Temperature (in degree celsius)')
    fig.update_layout(title = 'Temperature Variation in a Day (' + in_out_val + ')')
    fig.show()


# **Observations:** (assuming there are no outliers)<br>
# **(A) Inside room:**<br>
# 1. September month : <i>Maximum temperature variation</i> is observed in **morning & evening**.
# 2. October month : <i>Highest average (median) temperature</i> of **33&deg;C** has been recorded during **evening & night**.
# 3. <i>Lowest temperature</i> of <i>21&deg;C</i> is recorded in December month in the **morning**.
# 4. <i>Highest temperature</i> of <i>41&deg;C</i> is recorded in September month in the **afternoon** between 1400-1500 hours.
# 
# **(B) Outside room:**<br>
# 1. September month : <i>Maximum temperature variation</i> is observed during **afternoon**.
# 2. November month : <i>Highest average (median) temperature</i> of **42&deg;C** has been recorded in **morning**.
# 3. <i>Lowest temperature</i> of <i>24&deg;C</i> is recorded in September month in the **afternoon**.
# 4. <i>Highest temperature</i> of <i>51&deg;C</i> is recorded in September month in the **evening** between 1700-1800 hours.

# In[ ]:


tmp_df = round(temp_iot_data.groupby(['out_in', 'month', 'month_name', 'hour'])['temp'].mean(), 1).reset_index()
tmp_df.head()


# In[ ]:


for out_in_val in ['In', 'Out']:

    fig = go.Figure()
    
    for mth in range(9, 13):
    
        mth_name = pd.to_datetime('01' + str(mth) + '2019', format = '%d%m%Y').month_name()
        filter_cond = ((tmp_df['month'] == mth) & (tmp_df['out_in'] == out_in_val))

        fig.add_trace(go.Scatter(x = tmp_df.loc[filter_cond, 'hour'],
                                 y = tmp_df.loc[filter_cond, 'temp'],
                                 mode = 'lines+markers',
                                 name = mth_name))
    
    fig.update_xaxes(tickvals = list(range(0, 24)), ticktext = list(range(0, 24)), title = '24 Hours')
    fig.update_yaxes(title = 'Temperature (in degree Celsius)')
    fig.update_layout(title = 'Hourly Avg. Temperature for each month (' + out_in_val + ')')
    fig.show()


# **Observations:**<br>
# **(A) Inside room:**<br>
# 1. October month : Saw <i>Highest Average (Median) temperature</i> of **33.6&deg;C between 2000:2059 hours**.
# 2. December month : Saw <i>Lowest Average (Median) temperature</i> of **26.9&deg;C between 0400:0459 hours**.
# 
# **(B) Outside room:**<br>
# 1. October month : Saw the <i>Highest Average (Median) temperature</i> of **46.6&deg;C between 0800:0859 hours**.
# 2. December month : Saw the <i>Lowest Average (Median) temperature</i> of **29.4&deg;C between 1900:1959 hours**.
# 3. Compared to **September, October & November** months, December month's average temperature per hour has always been on lower side.

# In[ ]:


tmp_df = round(temp_iot_data.groupby(['out_in', 'month', 'month_name', 'day_name'])['temp'].mean(), 1).reset_index()
tmp_df.head()


# In[ ]:


for out_in_val in ['In', 'Out']:

    fig = go.Figure()
    
    for mth in range(9, 13):
    
        mth_name = pd.to_datetime('01' + str(mth) + '2019', format = '%d%m%Y').month_name()
        filter_cond = ((tmp_df['month'] == mth) & (tmp_df['out_in'] == out_in_val))

        fig.add_trace(go.Scatter(x = tmp_df.loc[filter_cond, 'day_name'],
                                 y = tmp_df.loc[filter_cond, 'temp'],
                                 mode = 'markers',
                                 name = mth_name,
                                 marker = dict(size = tmp_df.loc[filter_cond, 'temp'].tolist())                                 
                                ))
    
    fig.update_xaxes(title = 'Day', categoryarray = np.array(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']))
    fig.update_yaxes(title = 'Temperature (in degree Celsius)')
    fig.update_layout(title = 'Day-wise Avg. Temperature for each month (' + out_in_val + ')')
    fig.show()


# **Observation:** : No distinct patterns found that provides an evidence to prove that a relation exists between day of the week and rise or fall in temperature for each month.

# In[ ]:


tmp_df = temp_iot_data.groupby(['noted_date', 'out_in'])['temp'].mean().round(1).reset_index()


# In[ ]:


fig = go.Figure()

for out_in_val in ['In', 'Out']:

    filter_cond = (tmp_df['out_in'] == out_in_val)

    fig.add_trace(go.Scatter(x = tmp_df.loc[filter_cond, 'noted_date'],
                             y = tmp_df.loc[filter_cond, 'temp'],
                             mode = 'lines',
                             name = out_in_val))
    
fig.update_xaxes(title = 'Noted Date')
fig.update_yaxes(title = 'Temperature (in degree Celsius)')
fig.update_layout(title = 'Day-wise Temperature')
fig.show()


# **Observation:**<br>
# Upto mid of September, temperature wiggles around same range of values both inside and outside room. After that, outside temperature is higher compared to inside temperature.

# Request you to provide your valuable **feedback** on this kernel and kindly **upvote** if you like my work.
