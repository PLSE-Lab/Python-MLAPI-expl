#!/usr/bin/env python
# coding: utf-8

# 

# # Some Insights of Crime in Denver, Colorado
# 
# Hello guys, this is my first Kaggle submission so feel free to give me any critique or advice in my coding or insight if you guys have any. I will continue to work on this kernel to find new discoveries so nothing is final on this project. Thank you!
# 
# Before we start though, I do want to clarify what a district is in this dataset.
# 
# A police district is the geographical area that the police force of a certain police station regularly patrols. In this report, we will be focused on the different and amount of crimes that were reported in all the districts to find any important information.

# - <a href ='#0'>0. Preparing the Data</a>
# - <a href = '#1'>1. Total Reports of Each Districts</a>
# - <a href = '#2'>2. Districts vs Time</a>
# - <a href = '#3'>3. Crimes vs Time</a>
# - <a href ='#4'>4. Pie Charts of Crimes in Each District (2018)</a>
# - <a href ='#5'>5. Map Manipulation</a>
# 

# #  <a id = '0'>0. Preparing the Data</a>

# In[ ]:


import numpy as np
import pandas as pd
import folium
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from sklearn import linear_model
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
init_notebook_mode(connected = True)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
data = pd.read_csv("../input/crime.csv")
data.head()


# In[ ]:


data.shape


# The data used has 455872 number of reports and 19 columns of description for each report

# In[ ]:


data.info()


# From the information above, each column has a consistent number of non-null entries, with many equal or close to the total number of entries of the dataset, indicating that the dataset is clean. The only column that differs the most is the "Last Occurrence Date," however, this may be due to most reports being resolved when first reported.

# # <a id ='1'>1. Total Reports of Each District</a>

# ## **Which district has the most crime reports?**

# In[ ]:


district = data.DISTRICT_ID.value_counts().sort_index()
trace = go.Bar(x = district.index, y = district.values )
layout = go.Layout(title = 'Number of Reported Incidents By Police District',xaxis = dict(title='District ID', ticklen=10))
fig = go.Figure(data = [trace], layout = layout)
iplot(fig)


# From the graph above, we can see most crimes are reported in District 3 with 101k entries, following with District 6 with 90k entries.
# - District 7 has a significantly low number of reports compared to the other districts. When researched why, it is an airport.
# - I am assuming all the recording for each district started on the same date and each district has the same level of patrol for their area per land.

# # <a id='2'>2. Districts vs Time</a>

# ## **Did recording of each district start at the same time?**

# In[ ]:


data['REPORTED_DATE'] = pd.to_datetime(data['REPORTED_DATE'])
DID = data.DISTRICT_ID.unique()
DID.sort()
for x in DID:
    print("District: "+str(x)+"|"+str(min(data[data.DISTRICT_ID == x].REPORTED_DATE)))


# The info above displays the date and time of the first recorded entry of each district. From what is seen, all recordings of each district started at the same time, showing some consistency on how the data was recorded.

# ## **How has the number of reported crimes change over time?**
# 

# In[ ]:


DISTBYTIME = data.REPORTED_DATE.dt.date.value_counts().sort_index()
rollmean = DISTBYTIME.rolling(window = 100).mean()
rollstd = DISTBYTIME.rolling(window = 100).std()
trace = go.Scatter(x = DISTBYTIME.index, y = DISTBYTIME.values, mode = 'lines', name = 'Total')
tracemean = go.Scatter(x = DISTBYTIME.index, y=rollmean, mode = 'lines', name = 'Rolling Mean')
tracestd = go.Scatter(x = DISTBYTIME.index, y=rollstd, mode = 'lines', name = 'Rolling Std')
layout = go.Layout(title = 'Number of Reported Incidents vs Time', xaxis = dict(title = 'Date'))
fig = go.Figure(data = [trace, tracemean, tracestd], layout = layout)
iplot(fig)


# From the graph above, we can see some yearly cyclical pattern for the total number of incidents. It seems like there is a peak of total incidents during the summer, however, due to the fluctuation in the output, it is hard to notice. I did a rolling average with n = 100 (100 days) to make a more clear and smooth line that shows the pattern. Though the peaks might make it look like most crimes happen in the fall, remember, this line is a rolling average with n = 100. The peak is caused by the high values in the previous periods, thus caused by the high values during the summer.
# 
# Also, the number of incidents seems to stay consistent over time, however, there also looks like a small increasing trend in the data, in which I will create a quick linear regression line to visualize the trend.

# In[ ]:


from sklearn.linear_model import LinearRegression
data['Date'] = data.REPORTED_DATE.dt.date
data['count'] = 1;
total = data.groupby('Date').count()
total['Ticks'] = range(0,len(total.index.values))
lin = LinearRegression()
lin.fit(total[['Ticks']],total[['count']])
countpred = lin.predict(total[['Ticks']])
x = plt.plot(total[['Ticks']],countpred, color = 'black')
plt.xlabel('Days from 01-02-2014')
plt.ylabel('Number of Incidents')
plt.scatter(total[['Ticks']], total[['count']])


# Though it may be hard to notice, there is a small increasing trend in the number of incidents in Denver, Colorado.

# In[ ]:


trace = []
for x in range(1,8):
    SRC = data.loc[data.DISTRICT_ID == x, 'REPORTED_DATE'].dt.date.value_counts().sort_index()
    trace.append(go.Scatter(x = SRC.index, y = SRC.values, mode = 'lines', name = 'District '+str(x)))
    
layout = go.Layout(title = 'Number of Reported Incidents Separated by District vs Time', xaxis = dict(title = 'Date'))
fig = go.Figure(data = trace, layout = layout)
iplot(fig)


# When separating the data by districts, we continue to see that the number of incidents reported have been consistent over time. We can also see similar results as the graph in <a href='1'>Total Reports by District</a> of size of the number of crimes in each district relative to each other district.
# - **For those interested for why there is a spike on April 19, 2015 in District 6, there was a 2 day 4/20 rally that occured. I am not implying that it is the cause, however I wouldn't reject the idea. **

# In[ ]:


trace = []
rollmean = 0
for x in range(1,7):
    SRC = data.loc[data.DISTRICT_ID == x, 'REPORTED_DATE'].dt.date.value_counts().sort_index()
    rollmean = SRC.rolling(window = 30).mean()
    trace.append(go.Scatter(x = SRC.index, y = rollmean, mode = 'lines', name = 'District '+str(x)))
    
layout = go.Layout(title = 'Rolling Mean of Reported Incidents Separated by District vs Time', xaxis = dict(title = 'Date'))
fig = go.Figure(data = trace, layout = layout)
iplot(fig)


# *District 7 was filtered out due to its low values.*
# 
# Besides the peaks, we can see that the trends in each district stays stationary in the past 5 years.

# # <a id= '3'>3. Crimes vs Time</a>

# ## **What is the most occuring crime?**

# In[ ]:


data.OFFENSE_CATEGORY_ID.value_counts()


# From what can be seen above, most crime incidents are traffic accidents. The problem is that a traffic accident is not a crime unless it was a hit-or-run or an incident that validates an unlawful act.
# 
# ## From here on in the dataset, I will be filtering out all-other-crimes, traffic-accident, and other-crimes-against-persons to focus on non-traffic crimes and to get rid of of the grouped categories.

# In[ ]:


data = data[~(data.OFFENSE_CATEGORY_ID.isin(['all-other-crimes','traffic-accident','other-crimes-against-persons']))]
data.OFFENSE_CATEGORY_ID.value_counts()


# Now we can see that public disorder is the most reported, followed by larceny by a close margin.

# In[ ]:


num = 30
num1 = data[data.OFFENSE_CATEGORY_ID =='public-disorder']
crime1 = num1.REPORTED_DATE.dt.date.value_counts().sort_index()
rollmean1 = crime1.rolling(window = num).mean()
num2 = data[data.OFFENSE_CATEGORY_ID =='larceny']
crime2 = num2.REPORTED_DATE.dt.date.value_counts().sort_index()
rollmean2 = crime2.rolling(window = num).mean()
num3 = data[data.OFFENSE_CATEGORY_ID =='theft-from-motor-vehicle']
crime3 = num3.REPORTED_DATE.dt.date.value_counts().sort_index()
rollmean3 = crime3.rolling(window = num).mean()
num4 = data[data.OFFENSE_CATEGORY_ID =='drug-alcohol']
crime4 = num4.REPORTED_DATE.dt.date.value_counts().sort_index()
rollmean4 = crime4.rolling(window = num).mean()
num5 = data[data.OFFENSE_CATEGORY_ID =='auto-theft']
crime5 = num5.REPORTED_DATE.dt.date.value_counts().sort_index()
rollmean5 = crime5.rolling(window = num).mean()
num6 = data[data.OFFENSE_CATEGORY_ID =='burglary']
crime6 = num6.REPORTED_DATE.dt.date.value_counts().sort_index()
rollmean6 = crime6.rolling(window = num).mean()
num2.head()
trace1 = go.Scatter(x = crime1.index, y = crime1.values, mode = 'lines', name = 'Public Disorder')
trace2 = go.Scatter(x = crime2.index, y = crime2.values, mode = 'lines', name = 'Larcency')
trace3 = go.Scatter(x = crime3.index, y = crime3.values, mode = 'lines', name = 'Theft From Motor Vehicles')
trace4 = go.Scatter(x = crime4.index, y = crime4.values, mode = 'lines', name = 'Drug & Alcohol')
trace5 = go.Scatter(x = crime5.index, y = crime5.values, mode = 'lines', name = 'Auto Theft')
trace6 = go.Scatter(x = crime6.index, y = crime6.values, mode = 'lines', name = 'Burglary')
layout = go.Layout(title = 'Top 6 Crimes vs Time')
fig = go.Figure(data = [trace1,trace2,trace3, trace4,trace5, trace6], layout = layout)
iplot(fig)


# From the graph above, we can see some interesting thing like in the previous time series graphs:
# - A relatively stationary trend
# - Ouliers
# - Layering of the crimes

# In[ ]:


tracem1 = go.Scatter(x = crime1.index, y = rollmean1, mode = 'lines', name = 'Public Disorder')
tracem2 = go.Scatter(x = crime2.index, y = rollmean2, mode = 'lines', name = 'Larcency')
tracem3 = go.Scatter(x = crime3.index, y = rollmean3, mode = 'lines', name = 'Theft From Motor Vehicles')
tracem4 = go.Scatter(x = crime4.index, y = rollmean4, mode = 'lines', name = 'Drug & Alcohol')
tracem5 = go.Scatter(x = crime5.index, y = rollmean5, mode = 'lines', name = 'Auto Theft')
tracem6 = go.Scatter(x = crime6.index, y = rollmean6, mode = 'lines', name = 'Burglary')
layout = go.Layout(title = 'Top 6 Crimes Rolling Mean(n=30) vs Time')
fig = go.Figure(data = [tracem1,tracem2,tracem3, tracem4,tracem5, tracem6], layout = layout)
iplot(fig)


# Just like before, due to the fluctuation, it is difficult to notice any trends or patterns in the data. Therefore, I did a rolling average with n = 30 (30 days or 1 month) to smooth out the lines. Now we can see some of the cyclical patterns and trends for each crime.

# In[ ]:


num = 60
num1 = data[data.OFFENSE_CATEGORY_ID =='public-disorder']
crime1 = num1.REPORTED_DATE.dt.date.value_counts().sort_index()
rollmean1 = crime1.rolling(window = num).mean()
num2 = data[data.OFFENSE_CATEGORY_ID =='larceny']
crime2 = num2.REPORTED_DATE.dt.date.value_counts().sort_index()
rollmean2 = crime2.rolling(window = num).mean()
num3 = data[data.OFFENSE_CATEGORY_ID =='theft-from-motor-vehicle']
crime3 = num3.REPORTED_DATE.dt.date.value_counts().sort_index()
rollmean3 = crime3.rolling(window = num).mean()
num4 = data[data.OFFENSE_CATEGORY_ID =='drug-alcohol']
crime4 = num4.REPORTED_DATE.dt.date.value_counts().sort_index()
rollmean4 = crime4.rolling(window = num).mean()
num5 = data[data.OFFENSE_CATEGORY_ID =='auto-theft']
crime5 = num5.REPORTED_DATE.dt.date.value_counts().sort_index()
rollmean5 = crime5.rolling(window = num).mean()
num6 = data[data.OFFENSE_CATEGORY_ID =='burglary']
crime6 = num6.REPORTED_DATE.dt.date.value_counts().sort_index()
rollmean6 = crime6.rolling(window = num).mean()
num2.head()
trace1 = go.Scatter(x = crime1.index, y = rollmean1, mode = 'lines', name = 'Public Disorder')
trace2 = go.Scatter(x = crime2.index, y = rollmean2, mode = 'lines', name = 'Larcency')
trace3 = go.Scatter(x = crime3.index, y = rollmean3, mode = 'lines', name = 'Theft From Motor Vehicles')
trace4 = go.Scatter(x = crime4.index, y = rollmean4, mode = 'lines', name = 'Drug & Alcohol')
trace5 = go.Scatter(x = crime5.index, y = rollmean5, mode = 'lines', name = 'Auto Theft')
trace6 = go.Scatter(x = crime6.index, y = rollmean6, mode = 'lines', name = 'Burglary')
layout = go.Layout(title = 'Top 6 Crimes Rolling Mean(n=100) vs Time')
fig = go.Figure(data = [trace1,trace2,trace3, trace4,trace5, trace6], layout = layout)
iplot(fig)


# To make the trends and patterns more clearer to see, I made n = 60. 
# 
# **Some things to notice:**
# - Public disorder seems to be trending downwards over time while theft from motor vehicles is trending upwards.
# - Drugs and alcohol seems to have two peaks a year.

# # <a id = '4'>4. Pie Charts of Crimes for Each District (2018)</a>

# ### All-other-crimes, traffic-accident, and other-crimes-against-persons are still filtered out to focus on non-traffic crimes and to get rid of of the grouped categories. Also colors are based on the rankings, not the offense category in the pie charts below.

# In[ ]:


year2018 = (data.REPORTED_DATE >= '2018-01-01') & (data.REPORTED_DATE < '2019-01-01')
data = data[year2018]
for i in range(1,len(data.DISTRICT_ID.unique())+1):
    data1 = data[data.DISTRICT_ID == i]
    district1 = data1.OFFENSE_CATEGORY_ID.value_counts()
    labels = district1.index
    values = district1.values
    trace = go.Pie(labels = labels, values = values)
    name = 'District ' + str(i)
    layout = go.Layout(title = name)
    fig = go.Figure(data = [trace], layout = layout)
    iplot(fig)


# # <a id='5'>5. Map Manipulation</a>

# The map just shows an interactive map that shows the exact locations of the incidents. I filtered the data to only District 1 and public disorders as there is a limit on how much entries can be displayed to show in a notebook. Feel free to copy the code and change the filter options if interested in other categories. 

# In[ ]:


year2018 = (data.REPORTED_DATE >= '2018-01-01') & (data.REPORTED_DATE < '2019-01-01')
data2018 = data[(year2018) & (data.DISTRICT_ID == 1) & (data.OFFENSE_CATEGORY_ID =='public-disorder')]
idx = data2018['GEO_LAT'].isna() | data2018['GEO_LON'].isna()
data2018 = data2018[~idx]
m = folium.Map(location=[39.76,-105.02], tiles='Stamen Toner',zoom_start=13, control_scale=True)
from folium.plugins import MarkerCluster
mc = MarkerCluster()
for each in data2018.iterrows():
    mc.add_child(folium.Marker(location = [each[1]['GEO_LAT'],each[1]['GEO_LON']]))
m.add_child(mc)
display(m)


# In[ ]:




