#!/usr/bin/env python
# coding: utf-8

# Google has a wonderful tool for data viz: The Data Studio. I came to know about this tool when I was processing my data in Google Big Query. After processing the data they provide an option to visualise it using the data studio. There I found this litte wonder!
# 
# Its free open for everyone. You can connect using various options like csv files, google sheets, big query, jsons, gcp etc.
# I am seeing several dashborad already floating around on COVID-19. I thought give walkthough how you can create the below one embedded!

# In[ ]:


from IPython.display import IFrame
IFrame('https://datastudio.google.com/embed/reporting/bdd0af88-f4df-4357-a67b-550f9e7ad9c0/page/QQaIB', width='100%', height=900)


# Here's the standalone link: https://datastudio.google.com/reporting/bdd0af88-f4df-4357-a67b-550f9e7ad9c0

# You can see dashboard has several features:
#     1. The timeline feature: One select the time line by he/she wants to visulaize the data.
#     2. The Geo Viz. Clicking on which the whole dashboard get the country filter. You can aslo apply country/region filter from the side table.
#     3. Not only this you can drill though the map. Try going inside China. (Other countries don't the proper naming done in this data)
#     4. You can also chnage the metric in the map from confirmed cases to death and recovered cases using the optional metric.
#     5. In the botton I am showing the time trend. You can select any metric of your choice like Cumulative Confirmed cases or Daily cases or mortality rate. 
#     6. You can also apply timeline filter using this chart. And you can also apply country/region filter for this chart.

# **How I made this?**

# The challenging part is not making the dashboard. Its the preparation of the data required to be feed into the dashboard!
# Once you you prepare the data making dashboard is just like another BI tool.

# **Data Preparation**

# Key Points: 
# *     First I need to transform data from coloum format to to row format!  --- Pandas has a beautiful melt funtion that does this for me.
# *     I will need the daily count instead the cumulative count. I need to write code in such a way that I don't need to change anything unless the data source changes!
# *     Then I need to do same for all the metrics that I am creating. Like Cumulative, Daily and mortality rate, previous day cummulative, previous daily cumulative.
# 

# Lets Jump into the code!

# In[ ]:


#basic libraries
import pandas as pd
import numpy as np
import os
import gc
import datetime as dt
import math
from IPython.display import Image
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Directly pulling data from the source
path1 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
path2 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv"
path3 = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"
data_crfm = pd.read_csv(path1)
data_dead = pd.read_csv(path2)
data_reco = pd.read_csv(path3)


# In[ ]:


data_crfm.head()


# I found that data is little inconsistent. Cumulative counts have decreased for some countries/region (For example Japan you can see up on 23rd Jan their count decreased!). Which I tried correcting by below two funtions.
# I don't want to debate on this why this happened as this might not be relevant for this tutorial!

# In[ ]:


#What I am doing here is aloting the previous value to next date if the cumulative count is less on next date!
def modifier(x):
    return(x[0] if x[0]>x[1] else x[1])

def data_correctr(data):
    total_cols = data.shape[1]
    cols = data.columns
    for i in range(5,total_cols):
        data[cols[i]] = data[[cols[i-1], cols[i]]].apply(modifier, 1)
    return data


# In[ ]:


#getting corrected data set!
data_crfm_c = data_correctr(data_crfm)
data_dead_c = data_correctr(data_dead)
data_reco_c = data_correctr(data_reco)


# In[ ]:


total_cols = data_crfm_c.shape[1]

data_crfm_d = data_crfm_c.copy()
data_dead_d = data_dead_c.copy()
data_reco_d = data_reco_c.copy()

# this is done to calculate the percentage for every day (initalising day 1 to zero)
data_crfm_p = data_crfm_c.copy()
data_crfm_p.iloc[:,4] = 0
data_dead_p = data_dead_c.copy()
data_dead_p.iloc[:,4] = 0
data_reco_p = data_reco_c.copy()
data_reco_p.iloc[:,4] = 0


for i in range(5,total_cols):
    
    #converting cumulative to daily count
    data_crfm_d.iloc[:, i] = data_crfm_d.iloc[:, i] - data_crfm_c.iloc[:, i-1]
    data_dead_d.iloc[:, i] = data_dead_d.iloc[:, i] - data_dead_c.iloc[:, i-1]
    data_reco_d.iloc[:, i] = data_reco_d.iloc[:, i] - data_reco_c.iloc[:, i-1]
    
    #percentage change: I will store the previous day cumulative and apply percentage change later
    data_crfm_p.iloc[:, i] = data_crfm_c.iloc[:, i-1]
    data_dead_p.iloc[:, i] = data_dead_c.iloc[:, i-1]
    data_reco_p.iloc[:, i] = data_reco_c.iloc[:, i-1]

# Here I am storing previous day daily count I will need this to calculate percentage change metric: the 6 small box in the dashboard
data_crfm_dp = data_crfm_d.copy()  
data_crfm_dp.iloc[:,4] = 0
data_dead_dp = data_dead_d.copy()
data_dead_dp.iloc[:,4] = 0
data_reco_dp = data_reco_d.copy()
data_reco_dp.iloc[:,4] = 0

for i in range(5,total_cols):
    #percentage change: I will store the previous day daily and apply percentage change later
    data_crfm_dp.iloc[:, i] = data_crfm_d.iloc[:, i-1]
    data_dead_dp.iloc[:, i] = data_dead_d.iloc[:, i-1]
    data_reco_dp.iloc[:, i] = data_reco_d.iloc[:, i-1]


# In[ ]:


# Here comes the melt funtion of pandas. One line and your coloumns turns into rows!
df_crfm = pd.melt(data_crfm_d, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"Daily Confirmed"})
df_dead = pd.melt(data_dead_d, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"Daily Death"})
df_reco = pd.melt(data_reco_d, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"Daily Recovered"})

df_crfm_c = pd.melt(data_crfm_c, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"Cum Confirmed"})
df_dead_c = pd.melt(data_dead_c, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"Cum Death"})
df_reco_c = pd.melt(data_reco_c, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"Cum Recovered"})

df_crfm_p = pd.melt(data_crfm_p, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"PCum Confirmed"})
df_dead_p = pd.melt(data_dead_p, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"PCum Death"})
df_reco_p = pd.melt(data_reco_p, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"PCum Recovered"})

df_crfm_dp = pd.melt(data_crfm_dp, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"dPCum Confirmed"})
df_dead_dp = pd.melt(data_dead_dp, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"dPCum Death"})
df_reco_dp = pd.melt(data_reco_dp, id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long'], var_name = 'Time').rename(columns = {'value':"dPCum Recovered"})


# In[ ]:


df_crfm.head()


# In[ ]:


print(df_crfm.shape, df_dead.shape, df_reco.shape, df_crfm_c.shape, df_dead_c.shape, df_reco_c.shape, df_crfm_p.shape, df_dead_p.shape, df_reco_p.shape)


# In[ ]:


#Collecting the metric into 1 data frame
df = df_crfm.merge(df_dead[['Country/Region','Lat','Long', 'Time', 'Daily Death']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])
df = df.merge(df_reco[['Country/Region','Lat','Long', 'Time', 'Daily Recovered']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])

df = df.merge(df_crfm_c[['Country/Region','Lat','Long', 'Time', 'Cum Confirmed']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])
df = df.merge(df_dead_c[['Country/Region','Lat','Long', 'Time', 'Cum Death']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])
df = df.merge(df_reco_c[['Country/Region','Lat','Long', 'Time', 'Cum Recovered']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])

df = df.merge(df_crfm_p[['Country/Region','Lat','Long', 'Time', 'PCum Confirmed']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])
df = df.merge(df_dead_p[['Country/Region','Lat','Long', 'Time', 'PCum Death']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])
df = df.merge(df_reco_p[['Country/Region','Lat','Long', 'Time', 'PCum Recovered']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])

df = df.merge(df_crfm_dp[['Country/Region','Lat','Long', 'Time', 'dPCum Confirmed']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])
df = df.merge(df_dead_dp[['Country/Region','Lat','Long', 'Time', 'dPCum Death']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])
df = df.merge(df_reco_dp[['Country/Region','Lat','Long', 'Time', 'dPCum Recovered']], how = 'left', on = ['Country/Region','Lat', 'Long', 'Time'])


# In[ ]:


df.head()


# In[ ]:


#Some metrics

df['Mortality'] = ((df['Cum Death'] *100) /df['Cum Confirmed']).replace([np.inf, -np.inf], np.nan).fillna(0)
df['perchange Confirmed'] = ((df['Daily Confirmed']*100)/df['PCum Confirmed']).replace([np.inf, -np.inf], np.nan).fillna(0)
df['perchange Death'] = ((df['Daily Death']*100)/df['PCum Death']).replace([np.inf, -np.inf], np.nan).fillna(0)
df['perchange Recovered'] = ((df['Daily Recovered']*100)/df['PCum Recovered']).replace([np.inf, -np.inf], np.nan).fillna(0)


# In[ ]:


#last datatype corrections before feeding to data studio
df['Time'] = pd.to_datetime(df['Time']).dt.date
df['Lat Long'] = df['Lat'].astype(str)+","+df['Long'].astype(str)


# In[ ]:


df.to_csv("C:\\Users\\sikumar\\Downloads\\Corona_virus.csv", index = False)
print("Done!")


# Next Job!

# Either you can mannualy upload this data to google sheet or create an API to do this! 
# 
# Few tutorial for creating api:
# 
# 
#     https://developers.google.com/sheets/api/quickstart/js
#     
#     https://www.benlcollins.com/apps-script/api-tutorial-for-beginners/

# Google Data Studio!
# 
# Here you start! -> https://datastudio.google.com/u/0/

# In[ ]:


Image("../input/some-images-datastudio/Create new data source.PNG")


# **Step 1: Create a data source: Click + button on the top right corner and add a data source. You can connect using the google sheet or directly upload your csv file :)
# 
# 
# **

# In[ ]:


Image("../input/some-images-datastudio/Create new data source 2.PNG")


# **Step 2: Make sure the data type for your new data set is correct**

# In[ ]:


Image("../input/some-images-datastudio/Dtypes.PNG")


# **Step 3: Click on create a report!: You will be take to the report section where all the fun starts!

# In[ ]:


Image("../input/some-images-datastudio/workspace.PNG")


# Explore the tools. Most of them are self explainatory! 
# You get to draw your tools (filter, dropdown selection, apply basic formulae to the total count boxes and so on.. )
# 
# **Happy Dashboarding!**
# 
# **Save yourself world will save itself! :)**
