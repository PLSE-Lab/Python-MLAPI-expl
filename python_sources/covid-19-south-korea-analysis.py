#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Let's load all the datasets to see what data is present, and then explore couple of datasets for deeper insights. Since this data is specific to South Korea, let's see if there is any interesting pattern that can be found!

# In[ ]:


time_pr = pd.read_csv('/kaggle/input/coronavirusdataset/TimeProvince.csv')
pat_info = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')
time = pd.read_csv('/kaggle/input/coronavirusdataset/Time.csv')
weather = pd.read_csv('/kaggle/input/coronavirusdataset/Weather.csv')
region = pd.read_csv('/kaggle/input/coronavirusdataset/Region.csv')
time_age = pd.read_csv('/kaggle/input/coronavirusdataset/TimeAge.csv')
time_gndr = pd.read_csv('/kaggle/input/coronavirusdataset/TimeGender.csv')
policy = pd.read_csv('/kaggle/input/coronavirusdataset/Policy.csv')
pat_rte =  pd.read_csv('/kaggle/input/coronavirusdataset/PatientRoute.csv')
srch_trnd = pd.read_csv('/kaggle/input/coronavirusdataset/SearchTrend.csv')
case = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')
seoul = pd.read_csv('/kaggle/input/coronavirusdataset/SeoulFloating.csv')


# In[ ]:


# Create a deceased flag in the patient info file;

pat_info['deceased_flag']= np.where(pat_info['state'] == 'deceased',1,0)


# In[ ]:


death_rate = pat_info['deceased_flag'].value_counts()[1]*100/pat_info.shape[0]

print('death_rate :',round(death_rate,2),'%')


# Glad to see that South Korea has managed to keep the death rates very low, only 1.5%. Let's look at a pie chart of what kind of state each patient is in.

# In[ ]:


# What kind of state are the patients in?

agg = pat_info[['state','patient_id']].groupby('state').count().reset_index().sort_values(by =  'patient_id')

fig = go.Figure(data=[go.Bar(
            x= list(agg['patient_id']), 
            y=list(agg['state']),
            text=list(agg['patient_id']),
            textposition='auto',
            orientation ='h'
        )])

fig.update_layout(title = 'What kind of states are the patients in?',template='plotly_white')

fig.show()


# In[ ]:


state_agg = pat_info[['state','patient_id']].groupby('state').count().reset_index()

fig = go.Figure(data=[go.Pie(labels = list(state_agg['state'].values), values= list(state_agg['patient_id'].values), name='Distribution of State')])

fig.update_layout(title = 'What kind of states are the patients in?',template='plotly_white')

fig.show()


# Most of the patients seem to be released, which means they are being cured of COVID.. There is a good number of patients (41.8%) that are being isolated. South Korea seems to have taken good measures to isolate these patients and curb the spreading of the disease to other patients!
# 
# Should we also see which regions are these patients from? Who these isolated patients are? Are there any patients that are causing more spreading that SK should be worried about?

# In[ ]:


# We do notice that there are some patients that got affected through contact and others through overflow

agg = pat_info[['infection_case','patient_id']].groupby('infection_case').count().reset_index().sort_values(by = 'patient_id', ascending = False)[0:10]

agg.sort_values(by =  'patient_id',  inplace =True)

fig = go.Figure(data=[go.Bar(
            x= list(agg['patient_id']), 
            y=list(agg['infection_case']),
            text=list(agg['patient_id']),
            textposition='auto',
            orientation ='h'
        )])

fig.update_layout(title = 'Means of Case Transmission in South Korea',template='plotly_white')

fig.show()


# One thing we can clearly notice, most of the cases are transmitted through other patients, followed by others coming from overseas who are infected 

# In[ ]:


# Let's explore contact with patient

cwp = pat_info[pat_info['infection_case'] == 'contact with patient']
cwp.head()


# In[ ]:


# which patient infected most of the patients?

agg = cwp[['infected_by','patient_id']].groupby('infected_by').count().reset_index().sort_values(by = 'patient_id', ascending = False)[0:10]
#OMG, please be careful with patient 2000000205

agg.sort_values(by =  'patient_id',  inplace =True)

agg['infected_by'] = ['p_'+  str(i) for i in agg.infected_by.values]

fig = go.Figure(data=[go.Bar(
            x= list(agg['patient_id']), 
            y=list(agg['infected_by']),
            text=list(agg['patient_id']),
            textposition='auto',
            orientation ='h'
        )])

fig.update_layout(title = 'Which Patient infected the most?',template='plotly_white')

fig.show()


# We can notice that patient id'd 2000000205 has infected 51 others prior to being isolated or being diagnosed with COVID. 

# In[ ]:


# Who is this patient?

cwp[cwp['patient_id']== 2000000205]


# Good that South Korea has isolated this patient, else, this patient would have affected a lot many others!

# What kind of patients are causing contact with other patients and spreading infections?

# In[ ]:


cwp['dummy'] =1

gndr_agg = cwp[['sex','dummy']].groupby('sex').sum().reset_index()

age_agg = cwp[['age','dummy']].groupby('age').sum().reset_index()

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels = list(gndr_agg['sex'].values), values= list(gndr_agg['dummy'].values), name='sex'),
              1, 1)
fig.add_trace(go.Pie(labels = list(age_agg['age'].values), values= list(age_agg['dummy'].values), name='age'),
              1, 2)



# Most of the patients that were infected by contact are in the age group of 50's folled by 20's

# In[ ]:


# which regions have the most number of released vs isolated patients?

df= pat_info[['city','state','patient_id']].groupby(['city','state']).count().reset_index()

df = df.pivot_table(index = ['city'], columns = ['state'], values=['patient_id'])

df.fillna(0, inplace = True)

df.columns =  ['deceased','isolated','released']

df['Total'] = df.apply(lambda x :sum(x), axis = 1)

df.insert(0,'city',df.index.values)

df.reset_index(drop = True, inplace =  True)

#map the cities to latitudes and longitudes

df = df.merge(region,on = 'city')


# In[ ]:


# Let's create a bubble chart, with y axis as the total number, x axis being the name of the city and provinces, (two sub plots)

top_15_infec=  df.sort_values(by = 'Total',ascending = False)[0:15] 

# A bubble chart with bubble size the number of isolated?

# Create a text field for hovering over the plot;

hover_text = []

for index, row in top_15_infec.iterrows():
    hover_text.append(('City: {City}<br>'+
                       'Province: {Province}<br>'+
                       'Total: {Total}<br>'+
                       'Released: {Released}<br>'+
                       'Deceased: {Deceased}<br>'+
                       'Isolated: {Isolated}').format(City=row['city'],
                                            Province=row['province'],
                                            Total=row['Total'],
                                            Released=row['released'],
                                            Deceased=row['deceased'],
                                            Isolated=row['isolated']))

top_15_infec['text'] = hover_text

fig = go.Figure(data=[go.Scatter(
    x=list(top_15_infec['city']), 
    y=list(top_15_infec['Total']),
    mode='markers',
    marker=dict(
        color=[i for i in range(0,top_15_infec.shape[0])],
        size=list(top_15_infec['isolated']*0.5 + top_15_infec['deceased']*3),
        showscale  = True),
    
    text=top_15_infec['text']
)])

fig.update_layout(
    title='Cities with highest number of total cases in South Korea',
        template='plotly_white'

)


fig.show()


# One thing to clearly notice is Gyeongsan-si city has the highest number of cases, and also highest number of deaths of patients that 
# were affected by COVID. Seongnam-si is the second highest in the total number of cases, but beyond seongnam-si, deaths are showing 
# up for a very few cities.

# In[ ]:


# How about the provinces? Should we also check if there is a particular province of interest?

# Let's create a bubble chart, with y axis as the total number, x axis being the name of the city and provinces, (two sub plots)

province_df = df[['province','isolated','released','deceased','Total']].groupby('province').sum().reset_index().sort_values(by = 'Total', ascending = False)

top_15_infec=  province_df[0:15] 

# A bubble chart with bubble size the number of isolated?

# Create a text field for hovering over the plot;

hover_text = []

for index, row in top_15_infec.iterrows():
    hover_text.append(('Province: {Province}<br>'+
                       'Total: {Total}<br>'+
                       'Released: {Released}<br>'+
                       'Deceased: {Deceased}<br>'+
                       'Isolated: {Isolated}').format(Province=row['province'],
                                            Total=row['Total'],
                                            Released=row['released'],
                                            Deceased=row['deceased'],
                                            Isolated=row['isolated']))

top_15_infec['text'] = hover_text

fig = go.Figure(data=[go.Scatter(
    x=list(top_15_infec['province']), 
    y=list(top_15_infec['Total']),
    mode='markers',
    marker=dict(
        color=[i for i in range(0,top_15_infec.shape[0])],
        size=list(top_15_infec['isolated']*0.1 + top_15_infec['deceased']*0.5),
        showscale  = True),
    
    text=top_15_infec['text']
)])

fig.update_layout(
    title='Provinces with highest number of total cases in South Korea',
        template='plotly_white'

)


fig.show()


# Seoul province has the highest number, but most of the patients were released as they were cured
# Same with Gyeongsangbuk-do, but this province also has some good number of deaths
# Gyeonggi-do has the highest number of cases, but most of the patients have been isolated as they were thought to be dangerous to others. Be careful with Gyeonggi-do province for now.

# In[ ]:


# How many days were the patients held before being released?

from datetime import datetime

released = pat_info[pat_info['state'] == 'released']
released = released[released['released_date'].notna()]
released = released[released['confirmed_date'].notna()]


d2 = list(released['released_date'])

d1 = list(released['confirmed_date'])

n = len(d1)

diff = [abs(datetime.strptime(d2[i], "%Y-%m-%d")- datetime.strptime(d1[i], "%Y-%m-%d")).days  for  i  in range(0,n)]

released.insert(0,'diff',diff)


# In[ ]:


# Let's look at the distribution of number of days after which the patiets have been released

fig = go.Figure()

fig = make_subplots(rows=1, cols=2)

# Use x instead of y argument for horizontal plot
fig.add_trace(go.Box(y= released['diff'],name="Number of days for releasing patients"),1,1)
fig.add_trace(go.Histogram(x=released['diff'],histnorm='probability', name = 'distribution'),1,2)

fig.update_layout(title='Box and Histogram to show the number of days to cure and release patients',template='plotly_white')

fig.show()


# The average looks like around 24 days with some patients taking a long time to cure (around 120 days or 17 weeks)

# In[ ]:


# Let's look at the distribution of number of days after which the patiets have been released

fig = make_subplots(rows=1, cols=2)
# Use x instead of y argument for horizontal plot
fig.add_trace(go.Box(y= released['diff'][released['sex']=='male'],name="male"),1,1)
fig.add_trace(go.Box(y= released['diff'][released['sex']=='female'],name="female"),1,1)

fig.add_trace(go.Histogram(x=released['diff'][released['sex']=='male'],histnorm='probability', name = 'male hist'),1,2)
fig.add_trace(go.Histogram(x=released['diff'][released['sex']=='female'],histnorm='probability', name = 'female hist'),1,2)

fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)

fig.update_layout(title='Box to show the number of days for cure by gender',template='plotly_white')

fig.show()


# We look at the distributions and notice that male and female have the same distributions, but female patients have higher density compared to male patients in the histogram

# In[ ]:


# Let's look at the distribution of number of days after which the patiets have been released

fig = go.Figure()
# Use x instead of y argument for horizontal plot

ages = list(released['age'].drop_duplicates())

for i in ages:
    fig.add_trace(go.Box(y= released['diff'][released['age']==i],name=i))

fig.update_layout(title='Box to show the number of days by age group',template='plotly_white')

fig.show()


# In[ ]:


# what's the trend of infections over time in South Korea?

over_time = pat_info[['confirmed_date','state','patient_id']].groupby(['confirmed_date','state']).count().reset_index()

over_time = over_time.pivot_table(index = ['confirmed_date'], columns = ['state'], values=['patient_id'])

over_time.fillna(0, inplace = True)

over_time.columns = ['deceased','isolated','released']

over_time['Total'] = over_time.sum(axis = 1)

over_time.insert(0,'Date',over_time.index.values)


# In[ ]:


# Plot a time-series trend of total, deceased, isolated, released

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Get the list of countries

cols  = ['isolated','released','Total']

for i in cols:
    
    fig.add_trace(go.Scatter(x=over_time['Date'],y =over_time[i],mode='lines',name= i),secondary_y=False)

fig.add_trace(go.Scatter(x=over_time['Date'],y =over_time['deceased'],mode='lines',name= 'deceased'),secondary_y=True)

fig.update_layout(
    title='Trend of cases and state over time in SK',
        template='plotly_white'

)

fig.update_yaxes(title_text="Total, released, isolated", secondary_y=False)
fig.update_yaxes(title_text="deceased", secondary_y=True)



fig.show()


# March 2020 had recorded the highest number of deaths in South Korea
# The total number of cases for Covid along with isolations and deaths has gone down in May 2020
# We see a sudden increase in the number of cases post May 2020? Did South Korea implement any policy that would have resulted in this?
# SK has ramped up the number of isolations to avoid further spreading of COVID post May 2020  
# Were there any immigrants from COVID affected countries that travelled to SK in March -  April time frame that increased the number of cases?

# In[ ]:


# what kind of patients showed signs of death? This is cumulative deaths 

pted =  time_gndr.pivot_table(index = ['date'],columns =['sex'],values =['confirmed','deceased'])

pted.columns = ['confirmed_female','confirmed_male', 'deceased_female','deceased_male']

pted.insert(0,'date',pted.index.values)


# In[ ]:


# How does the search Trend look like, over time?
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Get the list of countries

cols  = ['confirmed_female','confirmed_male', 'deceased_female','deceased_male']

for i in cols:
    
    if i.startswith('confirmed'):
        
        cond = False
    else:
        cond = True
    
    fig.add_trace(go.Scatter(x=pted['date'],y =pted[i],mode='lines',name= i),secondary_y=cond)

fig.update_layout(
    title='Trend of cases and state over time in SK',
        template='plotly_white'

)

fig.update_yaxes(title_text="# cumulative confirmed cases over time by gender", secondary_y=False)

fig.update_yaxes(title_text="# cumulative deaths over time by gender", secondary_y=True)

fig.show()


# What do we notice?
# 
# The number of confirmations of COVID cases is high among female patients over male patients
# But, the death rate of male  patients is higher compared to female patients. If a male patient has been confirmed with COVID, he has a higher chance of death
# 

# In[ ]:


# How does the search Trend look like, over time?
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Get the list of countries

cols  = srch_trnd.columns[1:]

test = srch_trnd.copy()

test = test[test['date'] >= '2020-01-01']

for i in cols:
    
    fig.add_trace(go.Scatter(x=test['date'],y =test[i],mode='lines',name= i),secondary_y=False)

fig.update_layout(
    title='Trend of search about Corona in SK over time',
        template='plotly_white'

)

fig.update_yaxes(title_text=" SK Covid Numbers over time", secondary_y=False)



fig.show()


# Coronavirus was very popular in Q1 2020 in SK and was the most searched discussion topic. A lot of countries might have noticed teh same. 
# As people got to know more about it, they have started searching less and less

# # **> Please upvote if you liked the workbook!**
