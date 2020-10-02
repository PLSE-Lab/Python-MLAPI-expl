#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Welcome to my first work that is about global rates of suicides. 
#  
# In this kernel, I will analyze suicide rates data and try to visualize some data using seaborn and plotly.
# 
# Because of my first experience with data visualization, I would appreciate your feedbacks.

# 
# 
# **Content** : 
# 1. [Importing, Tidying and getting knowledge about data](#1)
# 1. [Visualization](#2):
#     1. [Per year](#3)
#     1. [By gender](#4)
#     1. [By generations](#5)   
#     1. [By age](#6)
#     1. [Top 10 countries](#7)   
#     1. [Gender ratios by countries](#8)    
#     1. [Correlations](#9)
#     1. [Correlation between GDP per capital and suicides](#10)
# 1. [Conclusion](#11)    

# <a id="1"></a> <br>
# # Importing, Tidying and getting knowledge about data

# In[ ]:


pip install chart-studio


# In[ ]:


# importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import os
print(os.listdir("../input"))


# In[ ]:


data = pd.read_csv('../input/master.csv')


# First, we will look at the data, try to tidy a bit and get some information about it.

# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


#I want to change the columns names for making it easier to code and understand.
data = data.rename(columns={'country':'Country','year':'Year','sex':'Gender','age':'Age',
                            'suicides_no':'SuicidesNo','population':'Population',
                            'suicides/100k pop':'Suicides_100kPop','country-year':'CountryYear',
                            'HDI for year':'HDIForYear',' gdp_for_year ($) ':'GdpForYearDollar',
                            'gdp_per_capita ($)':'GdpPerCapitalDollar','generation':'Generation'})


# In[ ]:


data.info()


# In[ ]:


data.GdpForYearDollar = data.GdpForYearDollar.str.replace(',','').astype('int64')


# In[ ]:


data.isnull().any()


# As you can see above there are missing values in HDIForYear column.

# In[ ]:


data.sample(5)


# Now we are good to visualize the data. 

# <a id="2"></a> <br>
# # Visualization
# 
# 

# <a id="3"></a> <br>
# * First, examine the number of suicides and associated with population ratio **per year **

# In[ ]:


yearList = data.Year.unique()
yearList.sort()
suicidesNumberPerYear = []
for i in yearList:
    total = sum(data.SuicidesNo[data.Year == i])
    suicidesNumberPerYear.append(total)
    
totalYearData = pd.DataFrame({"Year" : yearList, "NumberofSuicides" : suicidesNumberPerYear })
totalYearData.drop([31], axis = 0, inplace = True)

suicidesNumberPerYearper100k = []
for i in yearList:
    total = sum(data.Suicides_100kPop[data.Year == i])
    suicidesNumberPerYearper100k.append(total)
    

totalYearData2 = pd.DataFrame({"Year" : yearList, "NumberofSuicidesPer100k" : suicidesNumberPerYearper100k })
totalYearData2.drop([31], axis = 0, inplace = True)



f, (ax1,ax2) = plt.subplots(1,2,figsize = (20, 8))
sns.pointplot(x = totalYearData.Year, y = totalYearData.NumberofSuicides, color = "red", ax = ax1)
sns.pointplot(x = totalYearData2.Year, y = totalYearData2.NumberofSuicidesPer100k, color = "green", ax = ax2)
plt.subplot(ax1)
plt.xticks(rotation = 45)
plt.grid()
plt.subplot(ax2)
plt.xticks(rotation = 45)
plt.grid()
ax1.set_title("Number of Suicides per Year ")
ax2.set_title("Number of Suicides per Year (by pop) ")
plt.ylabel("")
plt.xlabel('Years')
plt.show()


# <a id="4"></a> <br>
# * Next, **gender ratio** of total number of suicides

# In[ ]:


gender = data.Gender.unique()
numSuicidesGender = []

for i in gender:
    total = sum(data.SuicidesNo[data.Gender == i])
    numSuicidesGender.append(total)
    
fig = go.Figure(data = [go.Pie(labels = list(gender),
                               values = numSuicidesGender)],
                layout = go.Layout(title = "Suicide Gender Rates"))

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=13, 
                  marker=dict(colors=["mediumturquoise","pink"],line=dict(color='#000000', width=1)))


fig.show()


# <a id="5"></a> <br>
# * Suicide rates by **generations**
# 
#     **Generations**:
# 
#     1. G.I. Generation -> 1900 - 1924
#     1. Silent -> 1925 - 1942
#     1. Boomers -> 1946 - 1964
#     1. Generation X -> 1965 - 1980
#     1. Millenials -> 1981 - 2000
#     1. Generation Z -> 2001 - 2010

# In[ ]:


generations = data.Generation.unique()
numSuicidesGeneration = []

for i in generations:
    total = sum(data.SuicidesNo[data.Generation == i])
    numSuicidesGeneration.append(total)
    
    
data1 = [go.Bar(
               x = generations,
               y = numSuicidesGeneration,
               name = "Number of Suicides",
               marker = dict(color = "rgba(116,173,209,0.8)",
                             line=dict(color='rgb(0,0,0)',width=1.0)))]
               

layout = go.Layout(xaxis= dict(title= 'Generations',ticklen= 5,zeroline= False))

fig = go.Figure(data = data1, layout = layout)

iplot(fig)


# <a id="6"></a> <br>
# * Suicide rates by **age** intervals

# In[ ]:


ageList = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years','75+ years']           
suicidesNoAge = []

for i in ageList:
    total = sum(data.SuicidesNo[data.Age == i])
    suicidesNoAge.append(total)
    
data2 = [go.Pie(values = suicidesNoAge,
               labels = ageList,
               hole = .3
                )]
layout = go.Layout( title = go.layout.Title(
                    text="Suicide by Ages",
                    x=0.47))

fig = go.Figure(data = data2, layout = layout)

iplot(fig)


# In[ ]:


suicidesNoAgeMale = []
suicidesNoAgeFemale = []
for i in ageList:
    total = sum(data.SuicidesNo[data.Gender == "male"][data.Age == i])
    suicidesNoAgeMale.append(total)
    
for i in ageList:
    total = sum(data.SuicidesNo[data.Gender == "female"][data.Age == i])
    suicidesNoAgeFemale.append(total)

trace1 = go.Bar(
                x = ageList,
                y = suicidesNoAgeMale,
                name = "Male")

trace2 = go.Bar(
                x = ageList,
                y = suicidesNoAgeFemale,
                name = "Female",
                xaxis = "x2",
                yaxis = "y2")

data3 = [trace1, trace2]

layout = dict(  xaxis=dict(domain=[0, 0.5]),
                yaxis=dict(domain=[0, 1]),
                xaxis2=dict(domain=[0.5, 1]),
                yaxis2=dict(domain=[0, 1]),
                title = "Suicide by Ages")
                 

fig = dict(data = data3, layout = layout) 

iplot(fig)


# <a id="7"></a> <br>
# * Countries having the highest rates of suicides 

# In[ ]:


countryList = data.Country.unique()
suicidesNoCountry = []
for i in countryList:
    total = sum(data.SuicidesNo[data.Country == i])
    suicidesNoCountry.append(total)
    
dfCountry = pd.DataFrame({"Country" : countryList, "noSuicides" : suicidesNoCountry})
newIndex = (dfCountry.noSuicides.sort_values(ascending = False)).index.values
sorted_dfCountry = dfCountry.reindex(newIndex)
topCountry = sorted_dfCountry.iloc[:10,:]
    
data4 = [go.Bar(x = topCountry.Country,
               y = topCountry.noSuicides,
               text = topCountry.Country,
               marker = dict(color = 'rgba(100, 200, 200, 0.6)',
                             line=dict(color='rgb(0,0,0)',width=1.5))
              )]

layout = go.Layout(title = "Top 10 countries",
                   hoverlabel = dict(bgcolor = "white"))

fig = go.Figure(data = data4, layout = layout)

iplot(fig)


# In[ ]:


Suicides_100kPopCountry = []

for i in countryList:
    total = sum(data.Suicides_100kPop[data.Country == i])
    Suicides_100kPopCountry.append(total)
    
dfCountry2 = pd.DataFrame({"Country" : countryList, "Suicides100k" : Suicides_100kPopCountry})
newIndex2 = (dfCountry2.Suicides100k.sort_values(ascending = False)).index.values
sorted_dfCountry2 = dfCountry2.reindex(newIndex2)
topCountry2 = sorted_dfCountry2.iloc[:10,:]

data5 = [go.Bar(x = topCountry2.Country,
               y = topCountry2.Suicides100k,
               text = topCountry2.Country,
               marker = dict(color = 'rgba(16, 112, 2, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5))
              )]

layout = go.Layout(title = "Top 10 countries by population",
                   hoverlabel = dict(bgcolor = "rgba(80, 26, 80, 0.7)"))

fig = go.Figure(data = data5, layout = layout)

iplot(fig)


# <a id="8"></a> <br>
# * Gender ratios of suicides by countries

# In[ ]:


suicidesCountryMale = []
suicidesCountryFemale = []

for i in countryList:
    total1 = sum(data.SuicidesNo[data.Gender == "male"][data.Country == i]) 
    total2 = sum(data.SuicidesNo[data.Gender == "female"][data.Country == i])
    try :
        mean1 = int((total1 / (total1 + total2)) * 100)
    except:
        pass
    try : 
        mean2 = int((total2 / (total1 + total2)) * 100)
    except:
        pass
    suicidesCountryMale.append(mean1)
    suicidesCountryFemale.append(mean2)

    
dfCountryGender = pd.DataFrame({"Country" : countryList, "Male" : suicidesCountryMale , "Female" : suicidesCountryFemale})
newIndex3 = (dfCountryGender.Male.sort_values(ascending = False)).index.values
sorted_dfCountryGender = dfCountryGender.reindex(newIndex3)
topCountryGender = sorted_dfCountryGender.iloc[:30,:]

newIndex4 = (dfCountryGender.Female.sort_values(ascending = False)).index.values
sorted_dfCountryGender2 = dfCountryGender.reindex(newIndex4)
topCountryGender2 = sorted_dfCountryGender2.iloc[:30,:]

    
trace1 = {
  'x': topCountryGender.Male, 
  'y': topCountryGender.Country,
  'name': 'Male',
  'type': 'bar',
  'orientation' : 'h',
  'marker' : {'color' : 'rgba(10,10,200,0.7)'}
}
trace2 = {
  'x': topCountryGender.Female,
  'y': topCountryGender.Country,
  'name': 'Female',
  'type': 'bar',
  'orientation' : 'h',
  'marker' : {'color' : 'rgba(252, 185, 65, 0.8)'}
}
trace3 = {
  'x': topCountryGender2.Male,
  'y': topCountryGender2.Country,
  'name': 'Male',
  'type': 'bar',
  'orientation' : 'h',
  'xaxis' : 'x2',
  'yaxis' : 'y2',
  'marker' : {'color' : 'rgba(10,10,200,0.7)'}
}
trace4 = {
  'x': topCountryGender2.Female,
  'y': topCountryGender2.Country,  
  'name': 'Female',
  'type': 'bar',
  'orientation' : 'h', 
  'xaxis' : 'x2',
  'yaxis' : 'y2',
  'marker' : {'color' : 'rgba(252, 185, 65, 0.8)'}
}

data6 = [trace1, trace2, trace3, trace4]

layout = go.Layout(title = "Number of suicides by countries comparing gender",
              xaxis = dict(showgrid = False,
              domain = [0.08, 0.49]),
                   
              yaxis = dict(showgrid = False,
              domain = [0, 1]),
                   
              xaxis2 = dict(showgrid = False,
              domain = [0.59, 1]),
                  
              yaxis2 = dict(showticklabels = True,showgrid = False, anchor = "x2",
              domain = [0, 1]),
              barmode = "relative",
              height = 700,
              width = 1050)
              

fig = dict(data = data6, layout = layout)

iplot(fig)


# <a id="9"></a> <br>
# * **Corrrelations**

# In[ ]:


f,ax = plt.subplots(figsize=(12, 6))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


# <a id="10"></a> <br>
# * **Correlation** between GDP per capital and suicides

# In[ ]:


suicides100kpop = []
suicidesGDP = []
for i in countryList:
    total = int(sum(data.Suicides_100kPop[data.Country == i]) / len(data.Suicides_100kPop[data.Country == i]))
    suicides100kpop.append(total)
    
for i in countryList:
    total = sum(data.GdpPerCapitalDollar[data.Country == i])
    mean = int(total / len(data.GdpPerCapitalDollar[data.Country == i]))
    suicidesGDP.append(mean)
    
dfCountryGDP = dfCountry.copy()
dfCountryGDP["GDP"] = suicidesGDP
dfCountryGDP['Suicides_100kPop'] = suicides100kpop 

data7 = [go.Scatter(
                    x = dfCountryGDP.GDP,
                    y = dfCountryGDP.Suicides_100kPop,
                    mode = "markers",
                    name = "GDP",
                    marker = dict(color = 'rgba(83, 51, 237, 1)'),
                    text = [dfCountryGDP.GDP,dfCountryGDP.noSuicides])]

layout = dict(title = 'Correlation between GDP per capital and suicides',
              xaxis = dict(title= 'GDP($)',ticklen= 5,zeroline= False),
              yaxis = dict(title= 'Suicides per 100k',ticklen= 5,zeroline= False),
              width = 800)
              
fig = go.Figure(data = data7, layout = layout)
iplot(fig)


# <a id="11"></a> <br>
# # Conclusion
# 
# * To conclude, I tried to analze and visualize the data.
# 
# * I would be very grateful for your comments and feedbacks. Because I am new at this tech and I want to be more qualified with it. So I'm open for advices, just leave a comment 
# 
# * Almost forgot, if you like it consider the upvote button :)
