#!/usr/bin/env python
# coding: utf-8

# # Covid-19 outbreak South Korea - Interactive Data Exploration using Plotly
# 
# ## Context
# In this notebook we do data exploration and visualization on patients infected with Covid-19 in South Korea. Visualizations are produced with Plotly.
# 
# #### What is a coronavirus?
# Coronaviruses are a large family of viruses which may cause illness in animals or humans.  In humans, several coronaviruses are known to cause respiratory infections ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS) and Severe Acute Respiratory Syndrome (SARS). The most recently discovered coronavirus causes coronavirus disease COVID-19.
# 
# #### What is COVID-19?
# COVID-19 is the infectious disease caused by the most recently discovered coronavirus. This new virus and disease were unknown before the outbreak began in Wuhan, China, in December 2019.
# 
# __COVID-19 has infected more than 7000 people in South Korea.__
# KCDC (Korea Centers for Disease Control & Prevention) announces the information of COVID-19 quickly and transparently.
# We make a structured dataset based on the report materials of KCDC and local governments.
# 
# ## Motivation
# 
# Most data reports only show data regarding the total number of confirmed, recovered and deceased patients. Here, the dataset includes details about gender and age as well recovery date and deceased date. This aditional features allow us to get more detailed insights about the outbreak.
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import plotly
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
import matplotlib.pyplot as plt
import folium
import warnings
py.init_notebook_mode()
warnings.filterwarnings("ignore")

from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


# In[ ]:


#load data
patient_path = "../input/coronavirusdataset/PatientInfo.csv"
time_path = "../input/coronavirusdataset/Time.csv"
route_path = "../input/coronavirusdataset/PatientRoute.csv"

df_route = pd.read_csv(route_path)
df_all_cases = pd.read_csv(time_path)
df_patients = pd.read_csv(patient_path)


# ## Number of tests, confirmed cases, deceased and recovered patients

# In[ ]:





df_all_cases['date'] = pd.to_datetime(df_all_cases['date'])

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_all_cases['date'], y=df_all_cases['test'], fill='tozeroy',name='total tests')) # fill down to xaxis
fig.add_trace(go.Scatter(x=df_all_cases['date'], y=df_all_cases['negative'], fill='tozeroy',name='negative test')) # fill down to xaxis
fig.add_trace(go.Scatter(x=df_all_cases['date'], y=df_all_cases['confirmed'], fill='tozeroy',name='positive test')) # fill down to xaxis
fig.update_layout(
    title = "Covid19 tests",
    #xaxis_range = [0,5.2],
    #yaxis_range = [0,3],
    yaxis_title="number of cases",
    font=dict(
        family="Arial, monospace",
        size=15,
        color="#7f7f7f"
    )
)
py.iplot(fig)



fig = go.Figure()
fig.add_trace(go.Scatter(x=df_all_cases['date'], y=df_all_cases['released'], fill='tozeroy',name='released')) # fill down to xaxis
fig.add_trace(go.Scatter(x=df_all_cases['date'], y=df_all_cases['deceased'], fill='tozeroy',name='deceased')) # fill down to xaxis
fig.update_layout(
    title = "Released and deceased over time",
    #xaxis_range = [0,5.2],
    #yaxis_range = [0,3],
    yaxis_title="number of cases",
    font=dict(
        family="Arial, monospace",
        size=15,
        color="#7f7f7f"
    )
)
py.iplot(fig)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_all_cases['date'], y=np.round(100*df_all_cases['deceased']/df_all_cases['released'],2), fill='tozeroy',name='ratio')) # fill down to xaxis
fig.update_layout(
    title = "Ratio of deceased/released",
    #xaxis_range = [0,5.2],
    #yaxis_range = [0,3],
    yaxis_title="deceased/released %",
    font=dict(
        family="Arial, monospace",
        size=15,
        color="#7f7f7f"
    )
)
py.iplot(fig)










# In[ ]:



df_patients['Age'] = None
for i in range(df_patients.shape[0]):
    if df_patients.birth_year.index[i] in df_patients[df_patients.birth_year.notna()].index:
        if  df_patients.birth_year.iloc[i] !=' ':
            df_patients['Age'].iloc[i] = 2020 - float(df_patients.birth_year.iloc[i])

df_recovered = df_patients[df_patients['state']=='released']
df_deceased = df_patients[df_patients['state']=='deceased']


# In[ ]:


df_patients


# ### Infection reason
# 
# 

# In[ ]:




fig = px.pie( values=df_patients.groupby(['infection_case']).size().values,names=df_patients.groupby(['infection_case']).size().index)
fig.update_layout(
    title = "Possible infection reason",
    font=dict(
        family="Arial, monospace",
        size=15,
        color="#7f7f7f"
    )
    )   
    
py.iplot(fig)


# ## Demographics analysis
# 
# Demographics data is not available for the total number of confirmed cases. 
# 
# * For 1557 cases, age was reported. 
# * For 1821 cases, sex was reported
# 
# * The median age of reported cases is 45 years old (min 0; max 104). <br/>
# * 43.8% of confirmed cases are male and 56.2% are female. <br/>
# * The median age of confirmed cases is 48 for females and 40 for males. <br/>
# 
# 

# In[ ]:


fig = px.histogram(df_patients[df_patients.Age.notna()],x="Age",marginal="box",nbins=20)
fig.update_layout(
    title = "number of confirmed cases by age group",
    xaxis_title="Age",
    yaxis_title="number of cases",
    barmode="group",
    bargap=0.1,
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 10),
    font=dict(
        family="Arial, monospace",
        size=15,
        color="#7f7f7f"
    )
    )
py.iplot(fig)


fig = px.pie( values=df_patients.groupby(['sex']).size().values,names=df_patients.groupby(['sex']).size().index)
fig.update_layout(
    title = "Sex distribuition of confirmed cases",
    font=dict(
        family="Arial, monospace",
        size=15,
        color="#7f7f7f"
    )
    )   
    
py.iplot(fig)

df_patients_aux = df_patients[df_patients.Age.notna()]
df_patients_aux=df_patients_aux[df_patients_aux.sex.notna()]
#df_patients_aux=df_patients_aux.sex.notna()
fig = px.histogram(df_patients_aux,x="Age",color="sex",marginal="box",opacity=1,nbins=20)
fig.update_layout(
    title = "number of confirmed cases by age group and sex",
    xaxis_title="Age",
    yaxis_title="number of cases",
    barmode="group",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 10),
    font=dict(
        family="Arial, monospace",
        size=15,
        color="#7f7f7f"
    ))
py.iplot(fig)




# * Median age of recovered patients is 42, while the median age for deceased patients is 77. __Apparently, higher death rates are linked to older patients. Higher recovery rates, are linked to younger patients.__

# In[ ]:


df_deceased_and_recovered = pd.concat([df_deceased,df_recovered])
fig = px.histogram(df_deceased_and_recovered,x="Age",color="state",marginal="box",nbins=10)
fig.update_layout(
    title = "Recovered and deceased patients by age group",
    xaxis_title="Age",
    yaxis_title="number of cases",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 10),
    bargap=0.2,
    barmode="group",
    xaxis_range = [0,100],
    font=dict(
        family="Arial, monospace",
        size=15,
        color="#7f7f7f"
    ))
py.iplot(fig)


# * Median age of recovered patients of female gender is 45, and 36 in males.
# * Median age of deceased patients of female gender is 85, and 73 in males.

# In[ ]:



df_deceased.drop(index=df_deceased[df_deceased.Age.isna()].index,inplace=True)
df_deceased.drop(index=df_deceased[df_deceased.sex.isna()].index,inplace=True)


df_recovered.drop(index=df_recovered[df_recovered.Age.isna()].index,inplace=True)
df_recovered.drop(index=df_recovered[df_recovered.sex.isna()].index,inplace=True)



# In[ ]:


fig = px.histogram(df_recovered,x="Age",color="sex",marginal="box",nbins=10)
fig.update_layout(
    title = "recovered patients by age and sex",
    xaxis_title="Age",
    yaxis_title="number of cases",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 10),
    bargap=0.2,
    barmode="group",
    xaxis_range = [0,100],
    font=dict(
        family="Arial, monospace",
        size=15,
        color="#7f7f7f"
    ))
py.iplot(fig)


fig = px.histogram(df_deceased,x="Age",color="sex",marginal="box",nbins=10)
fig.update_layout(
    title = "deceased patients by age and sex",
    xaxis_title="Age",
    yaxis_title="number of cases",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 10),
    bargap=0.2,
    barmode="group",
    xaxis_range = [0,100],
    font=dict(
        family="Arial, monospace",
        size=15,
        color="#7f7f7f"
    ))
py.iplot(fig)


# #### For cases with an outcome, we calculated the time between confirmation and the outcome (released/deceased). <br/>
# 
# * The median time after confirmation for recovery is 21 days.
# * The median time after confirmation for deceased cases is 5 days. 
# 
# This analysis is highly dependent on how much in advance the cases are detected.

# In[ ]:


df_deceased_and_recovered['time_lenght_to_recover_or_dead']= None
for i in range(df_deceased_and_recovered.shape[0]):
    if df_deceased_and_recovered['state'].iloc[i] == 'deceased':
        df_deceased_and_recovered['time_lenght_to_recover_or_dead'].iloc[i] = (pd.to_datetime(df_deceased_and_recovered['deceased_date'].iloc[i])- pd.to_datetime(df_deceased_and_recovered['confirmed_date'].iloc[i])).days
    if df_deceased_and_recovered['state'].iloc[i] == 'released':
        df_deceased_and_recovered['time_lenght_to_recover_or_dead'].iloc[i] = ( pd.to_datetime(df_deceased_and_recovered['released_date'].iloc[i]) - pd.to_datetime(df_deceased_and_recovered['confirmed_date'].iloc[i])).days
     


fig = px.histogram(df_deceased_and_recovered,x="time_lenght_to_recover_or_dead",color="state",marginal="box",nbins=10)
fig.update_layout(
    title = "Time do recovery/dead after confirmatition",
    xaxis_title="Days after confirmation",
    yaxis_title="number of cases",
    xaxis = dict(
        tickmode = 'linear',
        tick0 = -10,
        dtick = 5),
    bargap=0.2,
    barmode="group",
    xaxis_range = [-5,40],
    font=dict(
        family="Arial, monospace",
        size=15,
        color="#7f7f7f"
    ))
py.iplot(fig)


# ## Acknowledgements
# 
# * DS4C (Data Science for COVID-19) Project 
# * KCDC (Korea Centers for Disease Control & Prevention)
# 
# ## Data sources
# DS4C (Data Science for COVID-19) Project
# 
# To reprocess information provided by KCDC and local governments for easy data analysis
# To find meaningful patterns by applying various machine learning or visualization techniques
# Project Manager
# Jihoo Kim (datartist)
# Project Leader
# Seojin Jang (Seojin Jang)
# Seonghan Ryoo (incastle)
# Yeonjun In (Y.J)
# Project Engineer
# Kyeongwook Jang (Jeeu)
# Boyoung Song (bysong)
# Woncheol Lee (LeeWonCheol)
# Wansik Choi (wansik choi)
# Taehyeong Park (2468ab)
# Sangwook Park (Simon Park)
# Juhwan Park (JuHwan-Park)
# Minseok Jung (msjung)
# Youna Jung (You Na Jung)
# Logo Designer
# Rinchong Kim
# Github Repository
# 
# working with Big Leader and SK Telecom.
# sponsored by Google Korea (Soonson Kwon)
# 
# https://www.kaggle.com/kimjihoo/coronavirusdataset

# In[ ]:




