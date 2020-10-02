#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


# No. of Records , Attributes
data = pd.read_csv("../input/athlete_events.csv")
data.shape


# In[ ]:


# Glimpse of Dataset
data.head()


# In[ ]:


# checking for null objects
print("Missing values attributewise:".format(data.isnull().any()))
data.isnull().sum()


# In[ ]:


# data1: rows with all attribute details (No NaN values) (excluding Medal)
data1 = data[np.isfinite(data['Age'])]
data1 = data1[np.isfinite(data1['Weight'])]
data1 = data1[np.isfinite(data1['Height'])]
data1.describe()


# In[ ]:


meanAge = data1["Age"].mean()
meanHeight = data1["Height"].mean()
meanWeight = data1["Weight"].mean()
print(" Average Age of Athletes: ",meanAge)
print(" Average Height of Athletes: ",meanHeight)
print(" Average Weight of Athletes: ",meanWeight)


# In[ ]:


# Sorting data by year from 1896 to 2016
dataByYear= data.sort_values("Year")
dataByYear.head()


# In[ ]:


# Summer Olympic Data
data_summer = dataByYear[dataByYear.Season == "Summer"]
data_summer.head()


# In[ ]:


# Winter Olympic Data
data_winter = dataByYear[dataByYear.Season == "Winter"]
data_winter.head()


# In[ ]:


print("Total Editions:")
print("Summer:")
print(data_summer["Year"].unique())
print("Winter:")
print(data_winter["Year"].unique())


# In[ ]:


print("\nTotal Sports:\n", data["Sport"].unique())


# In[ ]:


print("Total Male & Female Participants:\n",data["Sex"].value_counts())


# In[ ]:


# Gold Medal Winners since 1896
gold_winners= data[data["Medal"]=="Gold"]
gold_winners.head()


# In[ ]:


# Silver Medal Winners since 1896
silver_winners= data[data["Medal"]=="Silver"]
silver_winners.head()


# In[ ]:


# Bronze Medal Winners since 1896
bronze_winners= data[data["Medal"]=="Bronze"]
bronze_winners.head()


# # Basic Visualization

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as mlt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls


# # Top Countries with Gold, Silver, Bronze

# In[ ]:


# Function to plot Medal Winners using Plotly
def medal_winners(med1, med2, med3):
    #mlt.subplots(figsize=(30,10))
    toppers1 = med1.groupby(['NOC']).size().reset_index(name='Count')
    toppers2 = med2.groupby(['NOC']).size().reset_index(name='Count')
    toppers3 = med3.groupby(['NOC']).size().reset_index(name='Count')
    cntr1 = toppers1.sort_values(by='Count', ascending=False)[:10]
    cntr2 = toppers2.sort_values(by='Count', ascending=False)[:10]
    cntr3 = toppers3.sort_values(by='Count', ascending=False)[:10]
      
    trace1 = go.Bar(x=cntr1.NOC, y=cntr1.Count)
    trace2 = go.Bar(x=cntr2.NOC, y=cntr2.Count, xaxis='x2', yaxis='y2')
    trace3 = go.Bar(x=cntr3.NOC, y=cntr3.Count, xaxis='x3', yaxis='y3')

    #fig = tls.make_subplots(rows=3, cols=1, subplot_titles=('Gold', 'Silver', 'Bronze'))
    dt = [trace1, trace2, trace3]
    layout = go.Layout(
        xaxis=dict(domain=[0, 0.3]),
        xaxis2=dict(domain=[0.33, 0.63]),
        xaxis3=dict(domain=[0.67, 1]),
        #yaxis1=dict(anchor='x1'),
        yaxis2=dict(anchor='x2'),
        yaxis3=dict(anchor='x3'),
        )
    fig = go.Figure(data=dt, layout=layout)
    #fig.append_trace(trace1, 1, 1)
    #fig.append_trace(trace2, 2, 1)
    #fig.append_trace(trace3, 3, 1)

    fig['layout'].update(title='Top Medal Winning Countries')
    fig['layout']['xaxis'].update(title='Top 10 Gold')
    fig['layout']['xaxis2'].update(title='Top 10 Silver')
    fig['layout']['xaxis3'].update(title='Top 10 Bronze')
    fig['layout']['yaxis'].update(title='Medal Count')

    py.plot(fig, filename='Top Medal Winning Countries')


# In[ ]:


# Plz Check output on Local machine
medal_winners(gold_winners, silver_winners, bronze_winners)


# # Top Athletes with Gold, Silver, Bronze

# In[ ]:


# Drop NaN values from Medal Column
data2 = data.dropna(subset=['Medal'])
data2.head()


# In[ ]:


# Save top 5 medal winners
gwa = gold_winners['Name'].value_counts().sort_values(ascending=True)[-5:]
swa = silver_winners['Name'].value_counts().sort_values(ascending=True)[-5:]
bwa = bronze_winners['Name'].value_counts().sort_values(ascending=True)[-5:]


# In[ ]:


# Plotting with Horizontal View
mlt.subplots(figsize=(18,24))
ax1 = mlt.subplot(311)
gwa.plot.barh(width=.7)
ax2 = mlt.subplot(312, sharex=ax1)
swa.plot.barh(width=.7)
ax3 = mlt.subplot(313, sharex=ax1)
bwa.plot.barh(width=.7)

ax3.set_xlabel('Medal Count')
ax1.set_ylabel('Athletes')
mlt.show()


# # Country Performance

# In[ ]:


# Function to Plot Country Performance in Summer or Winter Olympic 
def country_performance(cntr, edition):
    if edition=='Summer':
        data_cs = data_summer.groupby(['Year','NOC','Medal'])['NOC','Year','Medal'].size().reset_index(name='Count')
        data_cntr = data_cs[data_cs['NOC']==cntr]
        table = data_cntr.pivot_table(values='Count', index=['Year', 'NOC'], columns=['Medal'], aggfunc=np.sum)
        tab = table.reset_index()
        tab.plot.bar(x='Year', y=['Gold','Silver', 'Bronze'], color=["gold", "silver", "g"])
        mlt.xlabel('Editions')
        mlt.ylabel('Medals')
        fig=mlt.gcf()
        fig.set_size_inches(20,10)
        mlt.show()
    else:
        data_cw = data_winter.groupby(['Year','NOC','Medal'])['NOC','Year','Medal'].size().reset_index(name='Count')
        data_cntr = data_cw[data_cw['NOC']==cntr]
        table = data_cntr.pivot_table(values='Count', index=['Year', 'NOC'], columns=['Medal'], aggfunc=np.sum)
        tab = table.reset_index()
        tab.plot.bar(x='Year', y=['Gold','Silver', 'Bronze'], color=["gold", "silver", "g"])
        mlt.xlabel('Editions')
        mlt.ylabel('Medals')
        fig=mlt.gcf()
        fig.set_size_inches(20,10)
        mlt.show()


# In[ ]:


country_performance('USA','Summer')


# # Gender Participation

# In[ ]:


# Get Gender Count
sex_cnt= data_summer.groupby(['Year','Sex']).size().reset_index(name='Count')
sex_cnt.head()


# In[ ]:


# Plot Time Series using Seaborn
mlt.rcParams["axes.labelsize"] = 20
f, ax_s= mlt.subplots(figsize=(25,15))
#sns.set_context("notebook", font_scale=0.5, rc={"font.size":8,"axes.labelsize":5})

ax_s.set_title("Gender Participation in Summer Olympics",fontsize=20)

sns.pointplot(x="Year", y="Count", hue="Sex", data=sex_cnt, ax= ax_s)
mlt.show()


# In[ ]:


# Gender count in Winter Olympic
sex_cntw= data_winter.groupby(['Year','Sex']).size().reset_index(name='Count')
sex_cntw.head()


# In[ ]:


mlt.rcParams["axes.labelsize"] = 20
f, ax_s= mlt.subplots(figsize=(25,15))
#sns.set_context("notebook", font_scale=0.5, rc={"font.size":8,"axes.labelsize":5})

ax_s.set_title("Gender Participation in Winter Olympics",fontsize=20)

sns.pointplot(x="Year", y="Count", hue="Sex", data=sex_cntw, ax= ax_s)
mlt.show()


# In[ ]:




