#!/usr/bin/env python
# coding: utf-8

# #

# # Gun Violence Data
# 
# Comprehensive record of over 260k US gun violence incidents from 2013-2018
# 
# Content
# 
# This project aims to change that; we make a record of more than 260k gun violence incidents, with detailed information about each incident, available in CSV format. We hope that this will make it easier for data scientists and statisticians to study gun violence and make informed predictions about future trends.
# The CSV file contains data for all recorded gun violence incidents in the US between January 2013 and March 2018, inclusive.
# 
# Columns
# 
# incident_id  ID of the crime report 
# date             Date of crime 
# state                State of crime 
# city_or_county City/ County of crime 
# address Address of the location of the crime n_killed Number of people killed n_injured Number of people injured incident_url URL regarding the incident source_url Reference to the reporting source incident_url_fields_missing TRUE if the incident_url is present, FALSE otherwise congressional_district Congressional district id gun_stolen Status of guns involved in the crime (i.e. Unknown, Stolen, etc...) gun_type Typification of guns used in the crime incident_characteristics Characteristics of the incidence latitude Location of the incident location_description longitude Location of the incident n_guns_involved Number of guns involved in incident notes Additional information of the crime participant_age Age of participant(s) at the time of crime participant_age_group Age group of participant(s) at the time crime participant_gender Gender of participant(s) participant_name Name of participant(s) involved in crime participant_relationship Relationship of participant to other participant(s) participant_status Extent of harm done to the participant participant_type Type of participant sources state_house_district state_senate_district

# In[ ]:


import numpy as np


# In[ ]:


import pandas as pd


# In[ ]:


import os


# In[ ]:


import seaborn as sns


# In[ ]:


import matplotlib as mpl


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['xtick.labelsize']=8
plt.rcParams['ytick.labelsize']=8


# In[ ]:


import plotly


# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go


# In[ ]:


from plotly import tools


# In[ ]:


from plotly.offline import init_notebook_mode, iplot


# In[ ]:


from plotly.offline import plot


# In[ ]:


print(plotly.__version__)


# In[ ]:


import calendar


# In[ ]:


plt.style.use("seaborn")


# In[ ]:


plt.style.use("seaborn")


# In[ ]:


import heapq, string, os, random


# In[ ]:


from datetime import datetime


# In[ ]:


from wordcloud import WordCloud, STOPWORDS


# In[ ]:


from PIL import Image


# In[ ]:


import folium


# In[ ]:


from folium import plugins


# In[ ]:


from IPython.display import HTML, display


# In[ ]:


import collections 


# In[ ]:


from collections import Counter


# In[ ]:


GV=pd.read_csv("../input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv")


# # DATA Exploration

# In[ ]:


GV.info() # total usage


# In[ ]:



GV.memory_usage() # usage by column


# In[ ]:


GV.describe()


# In[ ]:


GV.head(5)


# In[ ]:


#Number of Rows
GV.shape[0]


# In[ ]:


#Number of Columns
GV.shape[1]


# In[ ]:


GV.index


# In[ ]:


GV.dtypes


# In[ ]:


#Column Names
GV.columns.values


# In[ ]:


GV.columns


# In[ ]:


GV['state'].value_counts()


# # Data Administration
# 
# Making Data easier to Read and more organized

# In[ ]:


#Arranging DateTime column into its component
GV['date']=pd.to_datetime(GV['date'])
GV.dtypes


# In[ ]:


GV['year'] = GV['date'].dt.year
GV['month'] = GV['date'].dt.month
GV['monthday'] = GV['date'].dt.day
GV['weekday'] = GV['date'].dt.weekday
GV.shape


# In[ ]:


GV['casualty']=GV['n_killed']+GV['n_injured']


# In[ ]:


#Segregating data Gender wise

GV["participant_gender"] = GV["participant_gender"].fillna("0::Unknown")
    
def gender(n) :                    
    gender_rows = []               
    gender_row = str(n).split("||")    
    for i in gender_row :              
        g_row = str(i).split("::")  
        if len(g_row) > 1 :         
            gender_rows.append(g_row[1])    

    return gender_rows

gender_series = GV.participant_gender.apply(gender)
GV["total_participant"] = gender_series.apply(lambda x: len(x))
GV["male_participant"] = gender_series.apply(lambda i: i.count("Male"))
GV["female_participant"] = gender_series.apply(lambda i: i.count("Female"))
GV["unknown_participant"] = gender_series.apply(lambda i: i.count("Unknown"))


# # Data Cleansing
# CLeaning and removing unnecessary Data

# In[ ]:


GV_null=GV.isnull().sum()


# In[ ]:


GV_dup=GV.duplicated().sum() # count of duplicates
GV_dup


# In[ ]:


GV_na=GV.isna().sum()


# In[ ]:


GV_nan=pd.concat([GV_null,GV_na],axis=1)
GV_nan


# In[ ]:


#Remove data not required 
GV.drop([
    "incident_url",
    "source_url",
    "incident_url_fields_missing",
    "sources"
], axis=1, inplace=True)


# In[ ]:


#year with maximum incidents recorded 
GV.year.value_counts().tail(10).plot(kind = 'bar', figsize = (15,15))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend('States')
plt.title('Gun Violence Incidents by year')
plt.ylabel('Number of incidents')
plt.xlabel('Year')


# In[ ]:


#Momth with  incidents recorded 
GV.month.value_counts().tail(10).plot(kind = 'bar', figsize = (15,15))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend('States')
plt.title('Gun Violence Incidents by year')
plt.ylabel('Number of incidents')
plt.xlabel('Month')


# In[ ]:


#weekday with incidents recorded 
GV.weekday.value_counts().tail(10).plot(kind = 'bar', figsize = (15,15))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend('States')
plt.title('Gun Violence Incidents by year')
plt.ylabel('Number of incidents')
plt.xlabel('Day')


# In[ ]:


#Total count Killed  Yearly due to Gun Violence

GV_yearly=GV.groupby(GV["year"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
GV_yearly_plot=sns.pointplot(x=GV_yearly.index, y=GV_yearly.No_Killed, data=GV_yearly,label="yearly_vs_killed")
GV_yearly


# In[ ]:


#Total count Killed  Monthly due to Gun Violence

GV_yearly=GV.groupby(GV["month"]).apply(lambda x: pd.Series(dict(No_Killed=x.n_killed.sum())))
GV_yearly_plot=sns.pointplot(x=GV_yearly.index, y=GV_yearly.No_Killed, data=GV_yearly,label="yearly_vs_killed")
GV_yearly


# In[ ]:


#Total count Injured  Yearly due to Gun Violence

GV_yearly=GV.groupby(GV["year"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
GV_yearly_plot=sns.pointplot(x=GV_yearly.index, y=GV_yearly.No_Injured, data=GV_yearly,label="yearly_vs_killed")
GV_yearly


# In[ ]:


#Total count Injured  Monthly due to Gun Violence

GV_yearly=GV.groupby(GV["month"]).apply(lambda x: pd.Series(dict(No_Injured=x.n_injured.sum())))
GV_yearly_plot=sns.pointplot(x=GV_yearly.index, y=GV_yearly.No_Injured, data=GV_yearly,label="yearly_vs_killed")
GV_yearly


# In[ ]:


import plotly
plotly.offline.init_notebook_mode() # run at the start of every ipython notebook
GV_cas = GV.reset_index().groupby(by=['state']).agg({'casualty':'sum', 'year':'count'}).rename(columns={'year':'count'})
GV_cas['state'] = GV_cas.index

trace1 = go.Bar(
    x=GV_cas['state'],
    y=GV_cas['count'],
    name='Number of Incidents',
)
trace2 = go.Bar(
    x=GV_cas['state'],
    y=GV_cas['casualty'],
    name='Total casualty',
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    margin=dict(b=150),
    legend=dict(dict(x=-.1, y=1.2)),
        )
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


#reference : https://github.com/amueller/word_cloud & https://github.com/amueller/word_cloud/blob/master/examples/masked.py

gun_mask = np.array(Image.open('../input/gungviol/gun_PNG1387.png'))
stopwords = set(STOPWORDS)
txt = " ".join(GV['gun_type'].dropna())
wc = WordCloud(mask=gun_mask, max_words=1200, stopwords=STOPWORDS, colormap='spring', background_color='Black').generate(txt)
plt.figure(figsize=(16,18))
plt.imshow(wc)
plt.axis('off')
plt.title('');


# In[ ]:


#reference : https://github.com/amueller/word_cloud & https://github.com/amueller/word_cloud/blob/master/examples/masked.py

gun_mask = np.array(Image.open("../input/usawcimg/USA-states (1).PNG"))
stopwords = set(STOPWORDS)
txt = " ".join(GV['location_description'].dropna())
wc = WordCloud(mask=gun_mask, max_words=1200, stopwords=STOPWORDS, colormap='spring', background_color='Black').generate(txt)
plt.figure(figsize=(16,18))
plt.imshow(wc)
plt.axis('off')
plt.title('');


# In[ ]:


GV['state'].value_counts().plot.pie(figsize=(20, 20), autopct='%.2f')
#Check for values to be displayed
plt.title("State wise pie diagram")
plt.ylabel('Number of State')


# In[ ]:


#  4. Statewise show dates with maximum incidents?- 

#Pie Chart
GV.state.value_counts().head().plot(kind = 'pie', figsize = (15,15))
plt.legend("state")
plt.title('Statewise distribution of incidents')
plt.xlabel('Number of incidents')
plt.ylabel('States')


# In[ ]:


#State with minimum incidents recorded 
GV.state.value_counts().tail(10).plot(kind = 'bar', figsize = (15,15))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend('States')
plt.title('Safest States in USA')
plt.ylabel('Number if incidents')
plt.xlabel('States')


# In[ ]:


state_s=pd.read_csv("../input/statesgv/states_GV.csv",index_col=0)


# In[ ]:


gun_killed = (GV[['state','n_killed']]
              .join(state_s, on='state')
              .groupby('Abbreviation')
              .sum()['n_killed']
             )


# In[ ]:


layout = dict(
        title = 'Safe State 2013-2018 ',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
data = [go.Choropleth(locationmode='USA-states',
             locations=gun_killed.index.values,
             text=gun_killed.index,
             z=gun_killed.values)]

fig = dict(data=data, layout=layout)

iplot(fig)


# In[ ]:


GV['guns'] = GV['n_guns_involved'].apply(lambda x : "5+" if x>=5 else str(x))

GV1 = GV['guns'].value_counts().reset_index()
GV1 = GV1[GV1['index'] != 'nan']
GV1 = GV1[GV1['index'] != '1.0']

labels = list(GV1['index'])
values = list(GV1['guns'])

trace1 = go.Pie(labels=labels, values=values, marker=dict(colors = ['#blueviolet', '#magenta', '#96D38C', '#cyan', '#lime', '#orangered', '#k', '#b', '#aquamarine']))
layout = dict(height=600, title='Number of Guns Used', legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)


# In[ ]:


GV_analysis = GV.sort_values(['casualty'], ascending=[False])
GV_analysis[['date', 'state', 'city_or_county', 'gun_type','n_killed', 'n_injured']].head(10)


# In[ ]:


Crimecount=GV['state'].value_counts().head(10)
Crimecount


# In[ ]:


plt.pie(Crimecount,labels=Crimecount.index,shadow=True)
plt.title("Top 10 High Crime Rate State")
plt.axis("equal")


# In[ ]:


state_D=pd.read_csv("../input/statessafe/states_D.csv",index_col=0)


# In[ ]:


gun_killed = (GV[['state','n_killed']]
              .join(state_D, on='state')
              .groupby('Abbreviation')
              .sum()['n_killed']
             )


# In[ ]:


layout = dict(
        title = 'To 10 Dangerous State 2013-2018 ',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
data = [go.Choropleth(locationmode='USA-states',
             locations=gun_killed.index.values,
             text=gun_killed.index,
             z=gun_killed.values)]

fig = dict(data=data, layout=layout)

iplot(fig)


# In[ ]:


#Ref : https://plot.ly/python/horizontal-bar-charts/

Types  = "||".join(GV['incident_characteristics'].dropna()).split("||")
incidents = Counter(Types).most_common(20)
inci1 = [x[0] for x in incidents]
inci2 = [x[1] for x in incidents]
trace1 = go.Scatter(
    x=inci2[::-2],
    y=inci1[::-2],
    name='Incident Report',
    marker=dict(color='rgba(50, 171, 96, 1.0)'),
    )
data = [trace1]
layout = go.Layout(
    barmode='overlay',
    margin=dict(l=350),
    width=900,
    height=600,
       title = 'Incident Report',
)

report = go.Figure(data=data, layout=layout)
iplot(report)


# In[ ]:


GV.state.value_counts().sort_index().plot(kind = 'barh', figsize = (20,15))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend('States')
plt.title('Statewise distribution of incidents')
plt.xlabel('Number of incidents')
plt.ylabel('States')


# In[ ]:


#State Vs No of People Killed
#no NAN in State and n_killed
sns.boxplot('state','n_killed',data=GV)
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=15)


# In[ ]:


#Violin plot analysis of number of killed and injured year wise
impact_numbers = GV[["n_killed","n_injured"]].groupby(GV["year"]).sum()
print(impact_numbers)
impact_numbers=sns.violinplot(data=impact_numbers,split=True,inner="quartile")


# In[ ]:


## Box Plot for n_killed or n_injured for Chicago data.
sns.boxplot("n_killed", "n_injured", data= GV)


# In[ ]:


df1 = GV.sort_values(['casualty'], ascending=[False])
df1[['date', 'state', 'city_or_county', 'address', 'n_killed', 'n_injured']].head(10)


# In[ ]:


#REf : https://pythonhow.com/web-mapping-with-python-and-folium/


map_GV=GV[GV['n_killed'] >= 3][['latitude', 'longitude', 'casualty', 'n_killed']].dropna()
m1 = folium.Map([39.50, -98.35], tiles='CartoDB dark_matter', zoom_start=3.5)
#m2 = folium.Map([39.50, -98.35], zoom_start=3.5, tiles='cartodbdark_matter')
markers=[]
for i, row in map_GV.iterrows():
    casualty = row['casualty']
    if row['casualty'] > 100:
        casualty = row['casualty']*0.1 
    folium.CircleMarker([float(row['latitude']), float(row['longitude'])], radius=float(casualty), color='#blue', fill=True).add_to(m1)
m1


# In[ ]:


#Number of person killed vs incident
sns.jointplot("incident_id",
             "n_killed",
             GV,
             kind="scatter",
             s=100, color="m",edgecolor="blue",linewidth=2)


# In[ ]:


#Swarm plot analysis of number of killed and number of guns involved
impact_numbers = GV[["n_killed","n_guns_involved"]].groupby(GV["year"]).sum()
print(impact_numbers)
impact_numbers=sns.swarmplot(x="n_killed",y="n_guns_involved",data=impact_numbers)


# In[ ]:


#Factor plot analysis of number of killed and injured year wise
impact_numbers = GV[["n_injured","n_guns_involved"]].groupby(GV["year"]).sum()
print(impact_numbers)
impact_numbers=sns.factorplot(data=impact_numbers,split=True,inner="quartile")


# In[ ]:


#Violin plot analysis of number of killed and injured year wise
impact_numbers = GV[["total_participant"]].groupby(GV["year"]).sum()
print(impact_numbers)
impact_numbers=sns.violinplot(data=impact_numbers,split=True)


# In[ ]:


# Plot using Seaborn
sns.lmplot(x='n_killed', y='n_injured', data=GV,
           fit_reg=False, 
           hue='state')
 
# Tweak using Matplotlib
plt.ylim(0, None)
plt.xlim(0, None)


# In[ ]:


#Number of person injured vs incident
sns.jointplot("incident_id",
             "n_injured",
             GV,
             kind="scatter",
             s=100, color="m",edgecolor="red",linewidth=2)


# In[ ]:


#Number of Male vs killed

sns.jointplot("male_participant",
             "n_killed",
             GV,
             kind="scatter",
             s=100, color="m",edgecolor="red",linewidth=2)


# In[ ]:


#Density plot for yearly incident 
yearly_casulaty = GV[["n_killed", "n_injured"]].groupby(GV["year"]).sum()
d_plot=sns.kdeplot(yearly_casulaty['n_killed'],shade=True,color="b")
d_plot=sns.kdeplot(yearly_casulaty['n_injured'],shade=True,color="g")
del(yearly_casulaty)


# In[ ]:


yearly_actor = GV[["total_participant","male_participant", "female_participant"]].groupby(GV["year"]).sum()
density_plot=sns.kdeplot(yearly_actor['total_participant'],shade=True,color="r")
density_plot=sns.kdeplot(yearly_actor['male_participant'],shade=True,color="g")
density_plot=sns.kdeplot(yearly_actor['female_participant'],shade=True,color="k")
del(yearly_actor)


# In[ ]:


sns.distplot(GV.year)


# In[ ]:


g = sns.FacetGrid(GV, col="year", col_wrap=4, ylim=(0, 10))
g.map(sns.pointplot, "male_participant", "n_killed", color=".3", ci=None);


# In[ ]:


g = sns.FacetGrid(GV, col="year", col_wrap=4, ylim=(0, 10))
g.map(sns.pointplot, "female_participant", "n_killed", color=".3", ci=None);


# In[ ]:


g = sns.FacetGrid(GV, col="year", col_wrap=4, ylim=(0, 10))
g.map(sns.pointplot, "unknown_participant", "n_killed", color=".3", ci=None);


# In[ ]:


sns.boxplot([GV.month, GV.n_injured])


# In[ ]:


sns.boxplot([GV.month, GV.n_killed])


# In[ ]:


g = sns.FacetGrid(GV, col="year", aspect=.5)
g.map(sns.barplot, "n_injured", "weekday")


# In[ ]:


g = sns.FacetGrid(GV, col="year", aspect=.5)
g.map(sns.barplot, "n_killed", "weekday")


# In[ ]:


g = sns.FacetGrid(GV, col="year")
g.map(plt.hist, "male_participant");


# In[ ]:


sns.set(style="ticks")
g = sns.FacetGrid(GV, row="n_killed", col="year", margin_titles=True)
g.map(sns.regplot, "year", "n_guns_involved", color=".3", fit_reg=False, x_jitter=.1);


# In[ ]:


g = sns.FacetGrid(GV, hue="year", palette="Set1", size=5, hue_kws={"marker": ["^", "v","*",">","<","+"]})
g.map(plt.scatter, "male_participant", "female_participant", s=100, linewidth=.10, edgecolor="black")
g.add_legend();


# # Conclusion
# 
# Most Incidient - Year - 2017
# Least Incident Year - 2013
# Most Incidient - Month- July
# Least Incident Month-  November
# Most Incidient - Week day - Saturday
# Least Incident Week day - Wednesday
# 
# Most Killed- Year - 2017
# Least Killed Year - 2013
# Most Killed- Month- January
# Least Killed Month-  April
# 
# Most injured Year - 2017
# Least injured Year - 2013
# Most injured Month- January
# Least injured Month-  Feburary
# 
# Top Three Dangerous State
# llinois
# California
# Florida
# 
# Top Three Safe State
# Hawaii
# Vermont
# Wyoming
# 
# Top Three Location where violence happens
# 
# Appartment
# Park
# High School
# 
# Number of Guns Used
# 
# 2 Guns - 59%
# 5+ Guns - 18.2%
# 3 Guns- 15.9%
# 4 Guns - 6.87%
# Top Three Gun Types recorded
# 
# Unknown
# Handgun
# Auto / Rifile
# 
# Top Three Incidents
# Shot -Dead (Murder, accidental & Sucide)
# Shots fired (No Injuries)
# Roberry with Injury / death
