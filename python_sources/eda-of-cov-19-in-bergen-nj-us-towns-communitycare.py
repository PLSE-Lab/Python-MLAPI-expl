#!/usr/bin/env python
# coding: utf-8

# # EDA of cov-19 in Bergen,NJ,US towns
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Bergen_County%2C_NJ_municipalities_labeled.svg/641px-Bergen_County%2C_NJ_municipalities_labeled.svg.png" width=600></img>
# 
# # Below is an Exploratory Data Analysis of covid cases in the municipalities of Bergen,NJ county in USA. The county has population of roughly a million. As of April 1st there are roughly 3500 confirmed cases and around 90 confirmed deaths.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import plotly.offline as ply
import plotly.graph_objs as go
ply.init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### I was looking for number of deaths due to cov-19 in Bergen NJ. And I found it in two sources - John hopkins and New york times. New york one seems more useful, so i used it for below analysis.

# In[ ]:


covid_data_world_daily_nytimes=pd.read_csv('/kaggle/input/us-counties-covid-19-dataset/us-counties.csv')
covid_data_world_daily_nytimes[covid_data_world_daily_nytimes.fips==34003].tail(18)


# In[ ]:


df_covid_deaths=covid_data_world_daily_nytimes[covid_data_world_daily_nytimes.fips==34003][['date','deaths']].tail(18).reset_index(drop=True).copy()
df_covid_deaths['New Daily Deaths']=df_covid_deaths.deaths.diff()
df_covid_deaths.tail()


# In[ ]:


df_demographics=pd.read_excel('/kaggle/input/bergen-nj-usa-county-detail/Bengen_county_Demographics.xlsx')
print(df_demographics.shape)
df_demographics.head()


# In[ ]:


# df_demographics['Population']=df_demographics['Population'].str.replace(',','').astype(int)
df_demographics.sort_values(by='Municipality (with map key)').head()


# #### Bergen county seems to have population of 905,116 (About million).

# ## Analysis of number of deaths happening in the county due to covid-19

# ### Daily deaths

# In[ ]:


county_pop=905116.
df_covid_deaths['New Daily Deaths average']=df_covid_deaths['New Daily Deaths'].rolling(window=5).mean()
df_covid_deaths['New Daily Deaths norm']=100000*df_covid_deaths['New Daily Deaths']/county_pop
df_covid_deaths['New Daily Deaths norm average']=df_covid_deaths['New Daily Deaths norm'].rolling(window=5).mean()


# In[ ]:


fig = px.bar(df_covid_deaths, x='date', y='New Daily Deaths')

fig.add_traces(go.Scatter(x=df_covid_deaths['date'], y=df_covid_deaths['New Daily Deaths average'],
                          mode = 'lines',
                          marker_color='black',
                          name='moving average')
                          )
fig.show()


# ### Daily deaths in terms of, per 100,000 people in the county

# In[ ]:


fig = px.bar(df_covid_deaths, x='date', y='New Daily Deaths norm')

fig.add_traces(go.Scatter(x=df_covid_deaths['date'], y=df_covid_deaths['New Daily Deaths norm average'],
                          mode = 'lines',
                          marker_color='black',
                          name='moving average')
                          )
fig.show()


# ### Roughly at present deaths due to covid are happening at an average of 3 person per 100,000 people per day in Bergen county but the trend is going down which is good

# ## Further detailed analysis of various municipalities in Bergen county

# In[ ]:


df_covid_bergen_all = pd.read_csv('/kaggle/input/covid19-new-jersey-nj-local-dataset/Covid-19-NJ-Bergen-Municipality.csv', index_col='Date', parse_dates=['Date'])
df_covid_bergen_all['New Daily cases']=df_covid_bergen_all['Total Presumptive Positives'].diff()
print(df_covid_bergen_all.shape)
df_covid_bergen_all.tail()


# In[ ]:


fig = px.bar(df_covid_bergen_all, x=df_covid_bergen_all.index, y='New Daily cases')

fig.add_traces(go.Scatter(x=df_covid_bergen_all.index, y=df_covid_bergen_all['New Daily cases'].rolling(window=5).mean(),
                          mode = 'lines',
                          marker_color='black',
                          name='moving average')
                          )
fig.show()


# In[ ]:


fig = px.bar(df_covid_bergen_all, x=df_covid_bergen_all.index, y=10000*df_covid_bergen_all['New Daily cases']/county_pop)
fig.add_traces(go.Scatter(x=df_covid_bergen_all.index, y=(10000*df_covid_bergen_all['New Daily cases']/county_pop).rolling(window=5).mean(),
                          mode = 'lines',
                          marker_color='black',
                          name='moving average')
                          )
fig.show()


# ### Roughly on average there are 500 cases being confirmed per day in the county and 5 cases per 10000 people per day

# ## Adding few more columns - number of cases in each Municipality interms of per 1000 people(based on its population) and cumulative values. All with respect to current date (2020-04-03)

# In[ ]:


df1=df_covid_bergen_all.iloc[[0,9,10,11,12,13,14,15,16,17,18,19,20]].T.reset_index()
df1.columns=['Municipality','2020-03-14','2020-03-23','2020-03-24','2020-03-25','2020-03-26','2020-03-27','2020-03-28','2020-03-29','2020-03-30','2020-03-31','2020-04-01','2020-04-02','2020-04-03']

df2=df_demographics[['Municipality (with map key)','Population']].sort_values(by='Municipality (with map key)').reset_index(drop=True)
df2.columns=['Municipality','Population']
df_covid_bergen_all_stat=df2.merge(df1,on='Municipality').sort_values(by='2020-04-03',ascending=False)
df_covid_bergen_all_stat['case_per_1000pop']=1000*(df_covid_bergen_all_stat['2020-04-03']/df_covid_bergen_all_stat['Population'])
df_covid_bergen_all_stat['cumsum_ratio_pop']=100*(df_covid_bergen_all_stat['Population'].cumsum()/df_covid_bergen_all_stat['Population'].cumsum().max())
df_covid_bergen_all_stat['cumsum_ratio_case']=100*(df_covid_bergen_all_stat['2020-04-03'].cumsum()/df_covid_bergen_all_stat['2020-04-03'].cumsum().max())

df_covid_bergen_all_stat.head(10)


# ## Also adding mean percent change happening with respect to confirmed cases. Useful to have an idea of how fast the cases are rising

# In[ ]:


df_covid_bergen_all_stat.loc[df_covid_bergen_all_stat['Municipality']=='New Milford','2020-04-02']=115  #for some reason it was 13 which is odd. so i changed it mean value of day before and after it.


# In[ ]:


trend_days=['2020-03-23','2020-03-24','2020-03-25','2020-03-26','2020-03-27','2020-03-28','2020-03-29','2020-03-30','2020-03-31','2020-04-01','2020-04-02','2020-04-03']

df_covid_bergen_all_stat['mean_percent_change']=df_covid_bergen_all_stat[trend_days].T.pct_change().mean()
df_covid_bergen_all_stat.head(10)


# ### Looking at how fast the rise in cases are happening at top 10 Municipalities that have high cases 

# In[ ]:


fig = px.scatter(df_covid_bergen_all_stat.head(10),x='Population',y='mean_percent_change',hover_data=["2020-04-03"],color='Municipality',size='2020-04-03',size_max=40)
fig.update_layout(xaxis_type="log")
fig.show()


# #### Seems cases are rising faster in Lodi Municipality. My concern would be Hackensack as it has high population and more cases as of now, so more cases could potentially be added in next few days.	 

# In[ ]:



temp = df_covid_bergen_all_stat[trend_days].copy()
temp.index=df_covid_bergen_all_stat['Municipality']
temp.loc['New Milford']['2020-04-02']=116.0
temp.head(10)


# In[ ]:


import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline


threshold = 30
f = plt.figure(figsize=(10,12))
ax = f.add_subplot(111)
for i,Combined_Key in enumerate(temp.index):
#     x = 30
    if i<10:
        t = temp.loc[temp.index== Combined_Key].values[0]
        t = t[t>threshold]

        date = np.arange(0,len(t))
        xnew = np.linspace(date.min(), date.max(), 30)
        spl = make_interp_spline(date, t, k=1)  # type: BSpline
        power_smooth = spl(xnew)
        if Combined_Key != 'Teaneck':
            plt.plot(xnew,power_smooth,'-o',label = Combined_Key,linewidth =3, markevery=[-1])
plt.tick_params(labelsize = 14)        
plt.xticks(np.arange(0,30,7),[ "Day "+str(i) for i in range(30)][::7]) 

x = np.arange(0,14)
y = 2**(x/2+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "gray")
plt.annotate(".. every second day",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)

x = np.arange(0,14)
y = 2**(x/7+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "gray")
plt.annotate(".. every week",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)

x = np.arange(0,14)
y = 2**(x/30+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "gray")
plt.annotate(".. every month",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)


x = np.arange(0,14)
y = 2**(x/4+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "Red")
plt.annotate(".. every 4 days",(x[-3],y[-1]),color="Red",xycoords="data",fontsize=14,alpha = 0.8)

# plot Params
plt.xlabel("Days",fontsize=17)
plt.ylabel("Number of Confirmed Cases",fontsize=17)
# plt.title("Trend Comparison of Different Counties (confirmed) (except for counties in New York State)",fontsize=22)
plt.legend(loc = "upper left")
plt.yscale("log")
plt.grid(which="both")
plt.show()


# ### Let's also look at overall how the cases have risen across all individual municipalities in comparison to 10 days ago(2020-03-23)

# In[ ]:


fig = px.scatter(df_covid_bergen_all_stat,x='2020-03-23',y='2020-04-03',hover_data=["Municipality"],color='Municipality')
fig.update_layout(xaxis_type="log", yaxis_type="log")
fig.show()


# ### In general higher the starting number more confirmed cases are there at present. But there are also more variations. For example saddle brook and Ho-Ho-Kus both had 4 cases on 23rd march but former has 60 cases now and later has only have four cases. There could be many reasons - effective containment measures, population size etc 

# In[ ]:


fig = px.scatter(df_covid_bergen_all_stat,x='Population',y='2020-04-03',hover_data=["Municipality"],color='Municipality')
fig.update_layout(xaxis_type="log", yaxis_type="log")
fig.show()


# ## In general higher the population higher the number of confirmed cases.

# In[ ]:


fig = px.line(df_covid_bergen_all_stat, x=range(0,70), y='cumsum_ratio_case')

fig.add_traces(go.Scatter(x=list(range(0,70)), y=df_covid_bergen_all_stat['cumsum_ratio_pop'],
                          mode = 'lines',
                          marker_color='black',
                          name='cumsum_ratio_pop')
                          )
fig.show()


# ## Roughly 10 counties account for 50 percent of cases and they comprise about one-third of the population.

# In[ ]:


# plot.yscale('log')
fig = px.scatter(df_covid_bergen_all_stat,x='Population',y='case_per_1000pop',hover_data=["2020-04-03"],color='Municipality',size='2020-04-03',size_max=40)
fig.update_layout(xaxis_type="log", yaxis_type="log")
fig.show()


# In[ ]:


1000*df_covid_bergen_all['Total Presumptive Positives'].tail(1)/county_pop


# ## The counties that have more population also have more confrmed cases per 1000 people. Roughly 5 out of thousand people are affected as of now in the county.

# ### Let's find the (interesting) counties that have the highest number of cases

# In[ ]:


df_covid_bergen_all.head()


# In[ ]:


towns = set(df_covid_bergen_all.columns) - set(['Total Presumptive Positives', 'New Daily cases'])
# towns


# In[ ]:


df_covid_bergen_all.iloc[-1][towns].nlargest(10)


# In[ ]:


highest_towns = df_covid_bergen_all.iloc[-1][towns].nlargest(10).index
highest_towns


# ## Simple trendline of the cumulative positive cases in these towns

# In[ ]:


df_covid_bergen_all[highest_towns].plot(figsize=(20, 10))


# 
# #### Teaneck is definitely the leader in number of positive cases.
# 

# # Looking at Teaneck in more detail

# In[ ]:


100*df_covid_bergen_all['Teaneck'].tail(1)/39776.0


# # It seems roughly one in 100 people are affected in the town at present

# In[ ]:


fig = px.bar(df_covid_bergen_all['Teaneck'], x=df_covid_bergen_all .index, y='Teaneck')
fig.add_traces(go.Scatter(x=df_covid_bergen_all.index, y=df_covid_bergen_all['Teaneck'].rolling(window=5).mean(),
                          mode = 'lines',
                          marker_color='black',
                          name='moving average')
                          )
fig.show()


# ## Confirmed cases trend (wrt 1000 people)

# In[ ]:


fig = px.bar(1000*df_covid_bergen_all['Teaneck']/39776.0, x=df_covid_bergen_all .index, y='Teaneck')
fig.add_traces(go.Scatter(x=df_covid_bergen_all.index, y=(1000*df_covid_bergen_all['Teaneck']/39776.0).rolling(window=5).mean(),
                          mode = 'lines',
                          marker_color='black',
                          name='moving average')
                          )
fig.show()


# ### In twelve days number of cases has risen from 1 in 1000 to 1 in 100.

# ## New cases per day

# In[ ]:


fig = px.bar(df_covid_bergen_all['Teaneck'].diff(), x=df_covid_bergen_all .index, y='Teaneck')
fig.add_traces(go.Scatter(x=df_covid_bergen_all.index, y=df_covid_bergen_all['Teaneck'].diff().rolling(window=5).mean(),
                          mode = 'lines',
                          marker_color='black',
                          name='moving average')
                          )
fig.show()


# ## Roughly 28 new cases are being confirmed every day at present. 

# ### Moving average per 1000 person (window =5 days).

# In[ ]:


fig = px.bar(1000*(df_covid_bergen_all['Teaneck'].diff()/39776.0), x=df_covid_bergen_all .index, y='Teaneck')
fig.add_traces(go.Scatter(x=df_covid_bergen_all.index, y=(1000*(df_covid_bergen_all['Teaneck'].diff()/39776.0)).rolling(window=5).mean(),
                          mode = 'lines',
                          marker_color='black',
                          name='moving average')
                          )
fig.show()


# # At present it seems in four days for every 1000 people there are atleast three people (from moving average of around 0.75) being confirmed of coronavirus in Teaneck Municipality.
# 
# # Also at present,  for every two days atleast one person might become deceased due to coronavirus in Teaneck Municipality (though the data is not available). Even by considering at county level situation of roughly one death per 100000 population per day and teaneck having high confirmed cases and of population of around 37000. it seems more likely.
# 
# # Being population of only 37000 people and that atleast 1 in 100 are affected if just every neighbourhood (which could be roughly hundred people) in the town make sure to take care of each other and also make sure to contain the spread of the disease from the confirmed cases we can improve the situation much better.
# 

# In[ ]:


us_cities_lat_long=pd.read_csv('/kaggle/input/us-cities-lat-long/US_cities_lat_long.csv')
us_cities_lat_long.head()


# In[ ]:


bergen_lat_long=us_cities_lat_long[us_cities_lat_long.county_fips==34003][['city','lat','lng']].copy()
bergen_lat_long.columns=['Municipality','lat','lng']
bergen_lat_long.loc[3685]=['Teaneck',40.8932, -74.0117]
bergen_lat_long.tail()


# In[ ]:


# bergen_lat_long[bergen_lat_long.Municipality=='Teaneck']


# In[ ]:


df_covid_bergen_all_stat_ll=df_covid_bergen_all_stat.merge(bergen_lat_long,on='Municipality')
df_covid_bergen_all_stat_ll.head(2)


# In[ ]:


bergen_series=pd.DataFrame(temp.unstack()).reset_index().merge(bergen_lat_long,on='Municipality')
bergen_series.columns=['Date','Municipality','Confirmed','Latitude','Longitude']
bergen_series.head()


# In[ ]:


bergen_series.Municipality.unique()


# In[ ]:


import plotly.express as px
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)


# tmp = df.copy()
# tmp['Date'] = tmp['Date'].dt.strftime('%Y/%m/%d')
fig = px.scatter_geo(bergen_series,lat="Latitude", lon="Longitude", color='Confirmed', size='Confirmed', locationmode = 'USA-states',
                     hover_name="Municipality", scope='usa', animation_frame="Date",
                     color_continuous_scale=px.colors.diverging.curl,center={'lat':40.889, 'lon':-74.0461}, 
                     range_color=[0, max(bergen_series['Confirmed'])])
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)')
fig.show()


# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Bergen_County%2C_NJ_municipalities_labeled.svg/641px-Bergen_County%2C_NJ_municipalities_labeled.svg.png" width=600></img>

# In[ ]:


# px.set_mapbox_access_token(open(".mapbox_token").read())
# # fig = px.scatter_mapbox(bergen_series[bergen_series.Date=='2020-03-26'], lat="Latitude", lon="Longitude",     color="Confirmed", size="Confirmed",
# #                   color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)
# # fig.show()


# In[ ]:


# df_covid_bergen_all['Teaneck'].diff()


# In[ ]:


# 1000*(df_covid_bergen_all['Teaneck'].diff()/39776.0).rolling(window=5).mean()

