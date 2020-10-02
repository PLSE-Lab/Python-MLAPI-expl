#!/usr/bin/env python
# coding: utf-8

# ## A deep dive into US mass shooting
# This analysis focused on mass shooting events in the US from 2013-2018 ( described as 'mass shooting' in the 'incident characteristics' column) <br>
# [I/EDA of suspect & victim age and gender](#I) 
# - [Box plot of victim and suspect age](#fig1) 
# - [Histogram of victim and suspect age distribution](#fig2) 
# - [Pie chart of gender distribution](#fig3) 
# <br>
# 
# [II/Geographical analysis of mass shooting](#II)
# - [2013-2018 US mass shooting map](#II1)
# - [Mass shooting plot of 2016](#II2)
# <br>
# 
# [III/ Time series EDA of mass shooting incidents](#III)
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 


# In[3]:


# Import data 
df = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
df.head()


# ** Visualize missing values**

# In[40]:


# Visualize missing data
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[4]:


# Create year and year-month columns
df['year'] = df['date'].apply(lambda x:pd.to_datetime(x).year)
df['monthyear'] = df['date'].apply(lambda x:'-'.join(x.split('-')[0:2]))


# ## I/ EDA of suspect & victim age and gender <a class="anchor" id="I"></a>
# In order to characterize suspects and victims, I got their Ids from the 'participant type' column and then use these Ids to get their age and gender <br>
# from the 'participant_age' and 'participant_gender' respectively. <br>
# About 61% of data set have all of these information. <br>
# In general, male dominated in both suspect (93.2 %) and victim categories (71.4 %). The age ranges of both categories are similar with median of 24, though among victims, there is a more noticeable portion of children (age < 15).
# 

# In[5]:


# Calculate total casualties as n_injured + n_killed
df['casualties'] = df['n_injured'] + df['n_killed']
# Create a text column for the map
df['text'] = (df['year']).astype(str) +'<br>' + df['city_or_county'] + '<br> n_killed ' + (df['n_killed']).astype(str) + '<br> n_injured '+ (df['n_injured']).astype(str)
# Get the mass shooting incidents only
indices = []
df_charac = df.dropna(subset=['incident_characteristics'])
for i in range(df_charac.shape[0]):
    if 'Mass' in df_charac['incident_characteristics'].iloc[i]:
        indices.append(i)
df_mass = df_charac.iloc[indices]


# In[43]:


# Drop NAs in three columns 
df_massnona = df_mass.dropna(subset=['participant_type','participant_age','participant_gender'])
# Get suspect id from the 'participant_type' column
df_massnona['suspect_id'] = df_massnona['participant_type'].apply(lambda x:', '.join([i[0] for i in x.split('||') if 'Suspect' in i]))
# Get ages for suspects and victims
suspect_ages = []
victim_ages = []
for k in range(0,df_massnona.shape[0]):
    suspect_ages += [int(i.split(':')[-1]) for i in df_massnona['participant_age'].iloc[k].split('||') if i[0] in df_massnona['suspect_id'].iloc[k]]
    victim_ages += [int(i.split(':')[-1]) for i in df_massnona['participant_age'].iloc[k].split('||') if i[0] not in df_massnona['suspect_id'].iloc[k]]


# **Box plot of victim and suspect age** <a class="anchor" id="fig1"></a>

# In[44]:


plt.figure(figsize=(10,5))
plt.subplot(121)
pd.Series(suspect_ages).plot(kind='box',label='Suspect age',fontsize=14)
plt.subplot(122)
pd.Series(victim_ages).plot(kind='box',label='Victim age',fontsize=14)
print(np.median(suspect_ages))
print(np.median(victim_ages))


# **Histogram of victim and suspect age distribution** <a class="anchor" id="fig2"></a>

# In[45]:


pd.Series(suspect_ages).hist(bins=20,alpha=0.5,label='Suspect age')
pd.Series(victim_ages).hist(bins=20,alpha=0.5, label='Victim age')
plt.legend()
plt.title("Histograms of age distribution")


# In[46]:


# Get genders for suspects and victims
suspect_genders = []
victim_genders = []
for k in range(0,df_massnona.shape[0]):
    suspect_genders += [i.split(':')[-1] for i in df_massnona['participant_gender'].iloc[k].split('||') if i[0] in df_massnona['suspect_id'].iloc[k]]
    victim_genders += [i.split(':')[-1] for i in df_massnona['participant_gender'].iloc[k].split('||') if i[0] not in df_massnona['suspect_id'].iloc[k]]


# In[47]:


suspect_genders = pd.Series(suspect_genders).value_counts()
victim_genders = pd.Series(victim_genders).value_counts()


# **Pie chart of gender distribution of suspects** <a class="anchor" id="fig 3"></a>

# In[48]:


plt.pie(suspect_genders,explode=(0,0),labels=['Male','Female'],autopct='%1.1f%%',radius=1)
plt.title('Suspects gender distribution',fontsize=14)


# **Pie chart of gender distribution of victims**

# In[49]:


plt.pie(victim_genders[0:2],labels=['Male','Female'],autopct='%1.1f%%',radius=1)
plt.title('Victims gender distribution',fontsize=14)


# ## II/Geographical analysis of mass shooting <a class="anchor" id="II"></a>
# Mass shooting incidents are plotted according to latitude, longitude, and year. <br>
# Colors of circles indicate range of total casualties (n_killed + n_injured) and size of circles indicate number of people killed in the incidents.

# **2013-2018 US mass shooting map** <a class="anchor" id="II1"></a>

# In[16]:


# Plot all mass shooting from 2013-2018
limits = [(0,5),(5,10),(10,30),(30,103)]
colors = ["rgb(255,250,2)","rgb(0,116,217)","rgb(255,7,246)","rgb(255,65,54)"]
cities = []
scale = 10
for i in range(len(limits)):
    df_sub = df_mass[df_mass['casualties'].isin(limits[i])]
    city = dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df_sub['longitude'],
        lat = df_sub['latitude'],
        text = df_sub['text'],
        marker = dict(
            alpha=0.5,
            size = df_sub['n_killed']*scale,
            color = colors[i],
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(limits[i][0],limits[i][1]) )
    cities.append(city)

layout = dict(
        title = '2013-2018 US mass gun violence' ,
        showlegend = True,
        geo = dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

fig = dict( data=cities, layout=layout )
iplot( fig, validate=False, filename='d3-bubble-map-populations')


# In[17]:


# Plot map for each year
limits = [(0,5),(5,10),(10,30),(30,103)]
colors = ["rgb(255,250,2)","rgb(0,116,217)","rgb(255,7,246)","rgb(255,65,54)"]
cities = []
scale = 10
def mass_plot(df,year):
    """
    Function to plot map of mass shooting for each year
    df: data
    year
    """
    df_year=df[df['year']==year]
    for i in range(len(limits)):
        df_sub = df_year[df_year['casualties'].isin(limits[i])]
        city = dict(
            type = 'scattergeo',
            locationmode = 'USA-states',
            lon = df_sub['longitude'],
            lat = df_sub['latitude'],
            text = df_sub['text'],
            marker = dict(
                alpha=0.5,
                size = df_sub['n_killed']*scale,
                color = colors[i],
                line = dict(width=0.5, color='rgb(40,40,40)'),
                sizemode = 'area'
            ),
            name = '{0} - {1}'.format(limits[i][0],limits[i][1]) )
        cities.append(city)

    layout = dict(
            title = '%d US mass gun violence' %year,
            showlegend = True,
            
            geo = dict(
                scope='usa',
                projection=dict(type='albers usa'),
                showland = True,
                landcolor = 'rgb(217, 217, 217)',
                subunitwidth=1,
                countrywidth=1,
                subunitcolor="rgb(255, 255, 255)",
                countrycolor="rgb(255, 255, 255)"
            ),
        )

    fig = dict( data=cities, layout=layout )
    iplot( fig, validate=False, filename='d3-bubble-map-populations')


# ** Mass shooting plot of 2016**  <a class="anchor" id="II2"></a>
# <br>
# The biggest circle is the the Orlando nightclub shooting 

# In[18]:


mass_plot(df_mass,2016)


# ## III/ Time series EDA of mass shooting incidents <a class="anchor" id="III"></a>
# More to time series prediction to come...

# In[28]:


plt.figure(figsize=(12,5))
plt.subplot(121)
sns.countplot(df['year'])
plt.title('Number of gun violence incidents from 2013-2018',fontsize=14)
plt.subplot(122)
sns.countplot(df_mass['year'])
plt.title('Number of mass shooting from 2013-2018',fontsize=14)


# In[32]:


plt.figure(figsize=(12,6))
plt.subplot(121)
plt.title("Total killed from 2013-2018 mass shooting",fontsize=14)
plt.ylabel("People")
df_mass.groupby('year')['n_killed'].sum().plot()
plt.subplot(122)
plt.title("Total injured from 2013-2018 mass shooting",fontsize=14)
df_mass.groupby('year')['n_injured'].sum().plot()


# In[34]:


plt.figure(figsize=(12,6))
plt.subplot(121)
plt.title("Average killed from 2013-2018 mass shooting",fontsize=14)
plt.ylabel("People")
df_mass.groupby('year')['n_killed'].mean().plot()
plt.subplot(122)
plt.title("Average injured from 2013-2018 mass shooting",fontsize=14)
df_mass.groupby('year')['n_injured'].mean().plot()


# In[35]:


plt.figure(figsize=(10,5))
df_mass.groupby('monthyear')['incident_id'].count().plot()
plt.title("Monthly total # of mass shooting incidents")


# In[36]:


plt.figure(figsize=(10,6))
df_mass.groupby('monthyear')['n_killed'].sum().plot(label='Total n_killed')
df_mass.groupby('monthyear')['n_injured'].sum().plot(label='Total n_injured')
plt.legend(loc='best')
plt.title("Monthly total of casualties")


# ### ARIMA model prediction of number of monthly mass shooting incidents

# In[16]:


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.stattools import adfuller


# In[12]:


def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis. Data is non-stationary ")


# In[6]:


# Get timeseries
ts = df_mass.groupby('monthyear')['incident_id'].count().reset_index()
ts.columns = ['month','total']
ts['month'] = pd.to_datetime(ts['month'])
ts.set_index('month',inplace=True)


# In[11]:


decomposition = seasonal_decompose(ts['total'], freq=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)


# In[13]:


adf_check(ts['total'])


# In[15]:


ts['first_difference'] = ts['total'] - ts['total'].shift(1)
adf_check(ts['first_difference'].dropna())


# In[17]:


model = sm.tsa.statespace.SARIMAX(ts['total'],order=(0,1,0), seasonal_order=(1,1,1,12))
results = model.fit()
print(results.summary())


# In[22]:


ts['forecast'] = results.predict(start = 40, end= 62, dynamic= True)  
ts[['total','forecast']].plot(figsize=(12,8))


# In[23]:


future_dates = [ts.index[-1] + DateOffset(months=x) for x in range(0,24) ]
future_dates_df = pd.DataFrame(index=future_dates[1:],columns=ts.columns)
future_df = pd.concat([ts,future_dates_df])
future_df.head()
future_df['forecast'] = results.predict(start = 62, end = 90, dynamic= True)  
future_df[['total', 'forecast']].plot(figsize=(12, 8)) 


# In[ ]:




