#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORTING LIBRARIES

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.request import urlopen
import json
from tqdm import tqdm
import warnings 

warnings.filterwarnings(action='ignore')


# # 1)  STEP 1. LOAD THE DATASETS

# In[ ]:


# load the datasets
train = pd.read_csv(r'/kaggle/input/covid19-global-forecasting-week-4/train.csv')
train['Date'] = pd.to_datetime(train['Date'])

test = pd.read_csv(r'/kaggle/input/covid19-global-forecasting-week-4/test.csv')
test['Date'] = pd.to_datetime(test['Date'])


submission = pd.read_csv(r'/kaggle/input/covid19-global-forecasting-week-4/submission.csv')


# # Submission data

# In[ ]:


submission.head(3)


# # Test data

# In[ ]:


test.head(3)


# # Train data

# In[ ]:


train.head()


# # 2) STEP 2. Some data exploration results
# ## Train dataset:
# * number of train dataset rows = 25040;
# * train start-end dates: 1) 2020-01-22; 2) 2020-04-10; (about 3 month dataset);
# * Total number of countries - 184;
# 
# ## Test dataset
# * number of test dataset rows = 13459;
# * test start-end dates: 1) 2020-04-02; 2) 2020-05-14;
# * Total number of countries - 184;
# 

# In[ ]:


# Lets take USA data
train_USA = train[train['Country_Region']=='US']

# grouped data sum
train_USA_grouped_sum = train_USA.groupby(by='Province_State').sum()[['ConfirmedCases', 'Fatalities']]
train_USA_grouped_sum = train_USA_grouped_sum.reset_index()
train_USA_grouped_sum['Death_Rate'] = train_USA_grouped_sum['Fatalities']/train_USA_grouped_sum['ConfirmedCases']*100
train_USA_grouped_sum = train_USA_grouped_sum.sort_values(by='Death_Rate', ascending=True)


# # 3) Lets plot Confirmed Cases and Fatalities properties over time for USA 

# In[ ]:


# CONFIRMED CASES PER EACH DAY CHANGE OVER TIME FOR USA; 
fig = px.line(train_USA, x="Date", y="ConfirmedCases", color="Province_State", title='CONFIRMED CASES PER EACH DAY CHANGE OVER TIME FOR USA')
fig.show()


# In[ ]:


# NUMBER OF FATALITIES PER EACH DAY OVER TIME FOR USA; 
fig = px.line(train_USA, x="Date", y="Fatalities", color="Province_State", title='NUMBER OF FATALITIES PER EACH DAY OVER TIME FOR USA')
fig.show()


# # 4) Total Number (Cumulative) of ConfirmedCases and Total Fatalities in USA

# In[ ]:


df_USA = train_USA.groupby('Date',as_index=False)['ConfirmedCases','Fatalities'].sum()
df_USA['ConfirmedCases_change'] = df_USA['ConfirmedCases'].shift(-1) - df_USA['ConfirmedCases']
df_USA['Fatalities_change'] = df_USA['Fatalities'].shift(-1) - df_USA['Fatalities']

fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_USA['Date'], y=df_USA['ConfirmedCases']),
    go.Bar(name='Deaths', x=df_USA['Date'], y=df_USA['Fatalities'])])

# Change the bar mode
fig.update_layout(barmode='overlay', title='Number of ConfirmedCases and Total Fatalities in USA')
fig.show()


# # 5) Total USA ConfirmedCases and Total Fatalities

# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='Cases', x=df_USA['Date'], y=df_USA['ConfirmedCases_change']),
    go.Bar(name='Deaths', x=df_USA['Date'], y=df_USA['Fatalities_change'])])

# Change the bar mode
fig.update_layout(barmode='overlay', title='Number of ConfirmedCases and Total Fatalities Change in USA')
fig.show()


# In[ ]:


train_USA.head(3)


# In[ ]:


train_USA['ConfirmedCases_shifted'] = train_USA.groupby(by='Province_State')['ConfirmedCases'].shift(-1)
train_USA['Fatalities_shifted'] = train_USA.groupby(by='Province_State')['Fatalities'].shift(-1)

def calc_percent_change(x1, x2):
    if x1==0:
        return 0
    else:
        return round(abs(x2-x1)/x1*100,2)
    
train_USA['ConfirmedCases_%_change'] = train_USA.apply(lambda x: calc_percent_change(x['ConfirmedCases'], x['ConfirmedCases_shifted']),axis=1 )
train_USA['Fatalities_%_change'] = train_USA.apply(lambda x: calc_percent_change(x['Fatalities'], x['Fatalities_shifted']),axis=1 )

train_USA['ConfirmedCases_delta'] = train_USA['ConfirmedCases_shifted'] - train_USA['ConfirmedCases']
train_USA['Fatalities_delta'] = train_USA['Fatalities_shifted'] - train_USA['Fatalities']


# # 6) CONFIRMED CASES CHANGE IN % TIME FOR USA; 

# In[ ]:


# CONFIRMED CASES CHANGE IN % TIME FOR USA; 
fig = px.line(train_USA, x="Date", y="ConfirmedCases_%_change", color="Province_State", title='CONFIRMED CASES PER EACH DAY  CHANGE OVER TIME IN % FOR USA')
fig.show()


# # 7) CONFIRMED CASES DELTA OVER TIME FOR USA; 

# In[ ]:


# CONFIRMED CASES DELTA OVER TIME FOR USA; 
fig = px.line(train_USA, x="Date", y="ConfirmedCases_delta", color="Province_State", title='CONFIRMED CASES DELTA FOR USA')
fig.show()


# # 8) Lets explore how ConfirmedCases changes per each State in detail

# In[ ]:


# CONFIRMED CASES DELTA OVER TIME FOR USA; 

for state in train_USA.Province_State.unique().tolist():
    fig = px.line(train_USA[train_USA['Province_State']==state], x="Date", y="ConfirmedCases_delta", color="Province_State", title=state, 
                 width=1000, height=300)
    fig.show()


# # 9) FATALITIES % CHANGE OVER TIME FOR USA; 

# In[ ]:


# FATALITIES % CHANGE OVER TIME FOR USA; 
fig = px.line(train_USA, x="Date", y="Fatalities_%_change", color="Province_State", title='FATALITIES % CHANGE OVER TIME FOR USA')
fig.show()


# # 10) FATALITIES %DELTA CHANGE OVER TIME FOR USA; 

# In[ ]:


# FATALITIES %DELTA CHANGE OVER TIME FOR USA; 
fig = px.line(train_USA, x="Date", y="Fatalities_delta", color="Province_State", title='FATALITIES %DELTA CHANGE OVER TIME FOR USA')
fig.show()


# In[ ]:


train_USA.head(3)


# # 11) CREATE PIE CHART FOR TOTAL CONFIRMED CASES AND FATALITIES IN USA 

# In[ ]:


# CREATE PIE CHART FOR TOTAL CONFIRMED CASES AND FATALITIES IN USA 
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(values=train_USA_grouped_sum['ConfirmedCases'], labels=train_USA_grouped_sum['Province_State'], name="ConfirmedCases"), 1, 1)
fig.add_trace(go.Pie(values=train_USA_grouped_sum['Fatalities'], labels=train_USA_grouped_sum['Province_State'], name="Fatalities"), 1, 2)
fig.update_layout(
    title_text="USA Total ConfirmedCases and Fatalities by Province for 2020-04-11")

fig.show()


# # 12) USA Death_Rate Horizontal Bar Chart by Province for 2020-04-11

# In[ ]:


fig = go.Figure(go.Bar(
            x=train_USA_grouped_sum['Death_Rate'],
            y=train_USA_grouped_sum['Province_State'],
            orientation='h'))
fig.update_layout(
    title_text="USA Death_Rate Horizontal Bar Chart by Province for 2020-04-11")
fig.show()


# In[ ]:


# Load data frame and tidy it.
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
df = df[['code', 'state']]
df = df.set_index('state')
df = df.to_dict()['code']
# converter to convert Province_States 


def dict_convert(row, dicti):
    if row in dicti.keys():
        return dicti[row]
    else:
        return None
    
    
train_USA_grouped_sum['Province_State_index'] = train_USA_grouped_sum['Province_State'].apply(lambda x: dict_convert(row=x, dicti=df))


# # 13) COVID USA CONFIRMED_CASES & FATALITIES MAPS

# In[ ]:


# COVID USA CONFIRMED_CASES MAP
fig = go.Figure(data=go.Choropleth(
    locations=train_USA_grouped_sum['Province_State_index'], # Spatial coordinates
    z = train_USA_grouped_sum['ConfirmedCases'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Blackbody',
    colorbar_title = "ConfirmedCases",
))

fig.update_layout(
    title_text = 'Covid USA ConfirmedCases',
    geo_scope='usa', # limite map scope to USA
)

fig.show()


# In[ ]:


# COVID USA FATALITIES MAP
fig = go.Figure(data=go.Choropleth(
    locations=train_USA_grouped_sum['Province_State_index'], # Spatial coordinates
    z = train_USA_grouped_sum['Fatalities'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Blackbody',
    colorbar_title = "Fatalities",
))

fig.update_layout(
    title_text = 'Covid USA Fatalities',
    geo_scope='usa', # limite map scope to USA
)

fig.show()


# # USA DATASET CONCLUSION
# 
# ## Top 5 Provinces with high ConfirmedCases:
# 1. New York;
# 2. New Jersey;
# 3. California; 
# 4. Michigan; 
# 5. Massachusetts.
# 
# 
# ## Top 5 Provinces with high Fatalities:
# 1. New York;
# 2. New Jersey;
# 3. Michigan; 
# 4. Louisiana; 
# 5. Washington.
# 
# ## Top 5 Provinces with high DeathRate
# 1. Washington; 
# 2. Kentucky;
# 3. Puerto Rico;
# 4. Vermont;
# 5. Michigan;
# 
# ## Top 5 Provinces with low DeathRate:
# 1. Wyoming;
# 2. Utah;
# 3. West Virginia;
# 4. Hawaii;
# 5. Virgin Islands.
# 
# 
# 1) Covid speed rate decreases over the time in percent change for each state (see **SECTIONS** 6,9) <br>
# 
# 2) Covid absolute number of ConfirmedCases and Fatalities increases (see **SECTIONS** 5). But ConfirmedCases & Fatalities speed decreases and the angle is almost equal to zero (Number of deaths and cases is almost the same) <br> 
# 
# 3) Covid Total ConfirmedCases and Fatalities Maps show that most cases are located in west (California) and north-east of USA (New-York, New-Jersey, Michigan).
# 

# ## To be Continued

# In[ ]:




