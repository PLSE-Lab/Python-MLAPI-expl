#!/usr/bin/env python
# coding: utf-8

# This data is drawn from my Strava archive. I started using a garmin in 2016, so will dig into three years worth of data: 2016, 2017, and 2018. A few rows of the raw data file is printed here. 

# In[ ]:


#Open libraries and read raw data
import os
import pandas as pd 
import numpy as np

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly as py
import plotly.graph_objs as go

from IPython.display import Image
data = pd.read_csv('../input/activities.csv', delimiter=',')
data.head(5)


# In[ ]:


data.info()


# The elapsed time and distance are in seconds and meters. I'd like to get the elapsed time into hours and minutes, and distance into miles. Below, I created those columns and left the raw data as is. I also separated the date into columns for year, month, and day. This will become useful later when I look at months over the three years. 

# In[ ]:


#Create hours, minutes_outof60, minutes, miles, and hours.minutes_outof60 columns
data['elapsed_time']=(data['elapsed_time']/60)/60
data['hours']=data['elapsed_time']//1
data['minutes_outof60']=(data['elapsed_time']%1).round(2)
data['minutes']=((data['elapsed_time']%1)*60).round(2)
data['hours.minutes_outof60']=data['elapsed_time']
data['elapsed_time']=(data['elapsed_time']*60)*60
# data['time_length']=data['hours']+data['minutes']
data['miles']=(data['distance']/1609.344).round(2)

#Separate date column into Date and Time
new = data["date"].str.split(" ", n = 1, expand = True) 
data["Date"]= new[0] 
data["Time"]= new[1] 

#Separate Date column into Year, Month, and Day.
new1=data['Date'].str.split("-",n=2,expand=True)
data['Year']=new1[0]
data['Month']=new1[1]
data['Day']=new1[2]

# #Change month number to name
# data.loc[ data['Month'] =='01', 'Month' ] = 'January'
# data.loc[ data['Month'] =='02', 'Month' ] = 'February'
# data.loc[ data['Month'] =='03', 'Month' ] = 'March'
# data.loc[ data['Month'] =='04', 'Month' ] = 'April'
# data.loc[ data['Month'] =='05', 'Month' ] = 'May'
# data.loc[ data['Month'] =='06', 'Month' ] = 'June'
# data.loc[ data['Month'] =='07', 'Month' ] = 'July'
# data.loc[ data['Month'] =='08', 'Month' ] = 'August'
# data.loc[ data['Month'] =='09', 'Month' ] = 'September'
# data.loc[ data['Month'] =='10', 'Month' ] = 'October'
# data.loc[ data['Month'] =='11', 'Month' ] = 'November'
# data.loc[ data['Month'] =='12', 'Month' ] = 'December'

data.tail(5)


# I've done one century. This is what I used to check my time and distance columns were printing okay.

# In[ ]:


data[data['miles']>100]


# So, for the first dive into visualizing some data I went for the most basic. Distance versus time. To create these, I created three datasets for each of the years. From this, I saw that I had a bunch of rides with elapsed time over 8 hours. These are due to commuting to work, pausing the ride, and starting it again 8+ hours later for the commute home. These are outliers so I removed them from this scatter plot. 

# In[ ]:


# Datasets for each year.
data_2018=data[data['Year']=='2018']
data_2017=data[data['Year']=='2017']
data_2016=data[data['Year']=='2016']
#Datasets for each year when the total elapsed time is less than 8 hours. 
data_2018_lt8hrs=data_2018[data_2018["hours.minutes_outof60"]<8]
data_2017_lt8hrs=data_2017[data_2017["hours.minutes_outof60"]<8]
data_2016_lt8hrs=data_2016[data_2016["hours.minutes_outof60"]<8]


# In[ ]:


data_2018_lt8hrs.head(5)


# From this scatter plot, I can see that 2017 was a year for good *and* bad mileage to time ratio. But, it's difficult to pick out other details from this plot due to the overlap. So, I'll keep going. 

# In[ ]:


#Miles vs. Hours.Minutes_outof60
mi_time=go.Figure()

mi_time.add_trace(go.Scatter(x=data_2016_lt8hrs["hours.minutes_outof60"], y=data_2016_lt8hrs['miles'],mode='markers',name='2016'))
mi_time.add_trace(go.Scatter(x=data_2017_lt8hrs["hours.minutes_outof60"], y=data_2017_lt8hrs['miles'],mode='markers',name='2017'))
mi_time.add_trace(go.Scatter(x=data_2018_lt8hrs["hours.minutes_outof60"], y=data_2018_lt8hrs['miles'],mode='markers',name='2018'))

mi_time.update_layout(title_text="Elapsed time versus mileage",
                      title_font_size=22, plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(title='Elapsed time',zeroline=True,zerolinecolor='#000000',zerolinewidth=0.7),
                      yaxis=dict( title='Mileage',zeroline=True,zerolinecolor='#000000',zerolinewidth=0.7))

py.offline.iplot(mi_time)


# This next plot will look at the sum of all miles for each month and compare it year over year. The dataframes I built for this plot are made below. 

# In[ ]:


#DATA FOR: Histogram with months on x-axis, mileage on y-axis, and different bars for each year. 
data_2016_lt8hrs_sumMi = data_2016_lt8hrs.groupby(['Month']) ['miles'].sum().reset_index()
data_2017_lt8hrs_sumMi = data_2017_lt8hrs.groupby(['Month']) ['miles'].sum().reset_index()
data_2018_lt8hrs_sumMi = data_2018_lt8hrs.groupby(['Month']) ['miles'].sum().reset_index()


# This plot shows masses. For one, I was fairly consistent with mileage during 2016, except for April and May. I believe I was overtrained during these months and took a lot of time off. 
# 
# The following year shows a strong start, but slightly less mileage than 2016. In March of 2017, I started my internship with Specialized and the mileage drops heavily after my first month of work. July was a solid month of distance covered, but in August and September, I tied up my internship, packed my bikes, and moved to England for grad school.  
# 
# Cycling took a big hit while I was at grad school. For the first few months, I went on the uni cycling club rides and trained for collegiate hill climb and track nationals. In January, though, that all changed. When the finals attacked. It also rained a lot and I was struggling to justify all the time I needed to spend cleaning my bike and changing my cruddy bike parts into working pieces again. Cheshire lanes are no joke... mud and grit for all. I'm happy to see I was able to get out more in February, but I do know that March and April were spent almost entirely on the rollers. Finals were in May and my sister's visit was in June. By July, I was working on my dissertation and this is when I realised the immense impact that going for coffee rides with my friends made upon me. Turns out, feeling happy makes working so much easier. If I can recommend, take a peak at the Penny-farthing cafe if you're ever around Knutsford and Tatton Park - it's a beautiful escape. School was done in September and cycling came back. 

# In[ ]:


#Histogram with months on x-axis, mileage on y-axis, and different bars for each year. 

numrides_mi=go.Figure()

numrides_mi.add_trace(go.Bar(x=data_2016_lt8hrs_sumMi["Month"], y=data_2016_lt8hrs_sumMi['miles'],name='2016'))
numrides_mi.add_trace(go.Bar(x=data_2017_lt8hrs_sumMi["Month"], y=data_2017_lt8hrs_sumMi['miles'],name='2017'))
numrides_mi.add_trace(go.Bar(x=data_2018_lt8hrs_sumMi["Month"], y=data_2018_lt8hrs_sumMi['miles'],name='2018'))

numrides_mi.update_layout(title_text="Mileage over time by Months and Years",
                      title_font_size=22, plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(type='category',title='Month',zeroline=True),
                      yaxis=dict(title='Mileage',zeroline=True,showline=True))

py.offline.iplot(numrides_mi)


# The next visualization I want to take a look at is the total number of rides I did each month.

# In[ ]:


#DATA FOR: Histogram with months on x-axis, number of rides on y-axis, and different bars for each year. 
data_2016_lt8hrs_Count = data_2016_lt8hrs.groupby(['Month']) ['id'].count().reset_index()
data_2017_lt8hrs_Count = data_2017_lt8hrs.groupby(['Month']) ['id'].count().reset_index()
data_2018_lt8hrs_Count = data_2018_lt8hrs.groupby(['Month']) ['id'].count().reset_index()


# Some months, I could have three rides per day if it was a race day. Warm up, race, cool down. Yep, gotta track it all. So, this plot doesn't necessarily show how many days each month that I went out on the bike. Stay tuned for that one. 
# 
# A lot of the same insights from mileage applies to this one. The major peak in March of 2017 is when I started commuting, and would start/stop morning and evening commutes, and rides during the day. 
# 
# Some of the sadder numbers to reflect grad school: 
# * in June 2018, I rode 8 times
# * in August 2018, I rode 4 times

# In[ ]:


#Histogram with months on x-axis, number of rides on y-axis, and different bars for each year. 
numrides_count=go.Figure()

numrides_count.add_trace(go.Bar(x=data_2016_lt8hrs_Count["Month"], y=data_2016_lt8hrs_Count['id'],name='2016'))
numrides_count.add_trace(go.Bar(x=data_2017_lt8hrs_Count["Month"], y=data_2017_lt8hrs_Count['id'],name='2017'))
numrides_count.add_trace(go.Bar(x=data_2018_lt8hrs_Count["Month"], y=data_2018_lt8hrs_Count['id'],name='2018'))

numrides_count.update_layout(title_text="Number of rides over time by Months and Years",
                      title_font_size=22, plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(type='category',title='Month',zeroline=True),
                      yaxis=dict(title='Number of rides',zeroline=True,showline=True))

py.offline.iplot(numrides_count)

