#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


covid_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid_data=covid_data.drop('SNo', axis=1) #dropping the SNo and Last Update columns
covid_data=covid_data.drop('Last Update', axis=1)
covid_data.info()

#converting the data type on each of the fields
covid_data["ObservationDate"] = covid_data['ObservationDate'].astype('datetime64')
covid_data["Confirmed"] = covid_data['Confirmed'].astype('int64')
covid_data["Deaths"] = covid_data['Deaths'].astype('int64')
covid_data["Recovered"] = covid_data['Recovered'].astype('int64')
#covid_data["Country/Region"] = covid_data['Country/Region'].astype('char')


# 

# In[ ]:


#Cleaning the covid_data file for analysis
covid_data=covid_data.rename(columns={"ObservationDate": "Date","Country/Region":"Country"})
covid_data_date=pd.DataFrame(covid_data.groupby(by='Date').sum())
covid_data_date['Date'] = covid_data_date.index
covid_data_date.Date = covid_data_date.Date.apply(lambda x: x.date())


# In[ ]:


#Total no.of cases by date
plt.figure(figsize=(15,15))
plt.xlabel('Date', fontsize=18)
plt.ylabel('Confirmed', fontsize=18)
plt.xticks(rotation=90)
Confirmed1=sns.barplot(x='Date',y='Confirmed',data=covid_data_date, color='blue')


# From the graph we can see that since March 13th, we see a significant rise in no.of confirmed cases. If we follow the news we can see that a lot of countries have initiated lock downs, workplaces have asked employees to work from home. Then why the increase in numbers. It could also be since a lot of countries are increasing the no.of tests they are conducting and that could be another reason why so many cases are getting confirmed.

# In[ ]:


#No.of Deaths by date
plt.figure(figsize=(15,15))
plt.xlabel('Date', fontsize=18)
plt.ylabel('Deaths', fontsize=18)
plt.xticks(rotation=90)
Deaths1=sns.barplot(x='Date',y='Deaths',data=covid_data_date, color='red')


# As of 13th March, we can see a significant rise in the death rate as compared to the previous dates. We should investigate futher and see why this sharp rise occured. 

# In[ ]:


#No.of Recoveries by date
plt.figure(figsize=(15,15))
plt.xlabel('Date', fontsize=18)
plt.ylabel('Recovered', fontsize=18)
plt.xticks(rotation=90)
Recovered1=sns.barplot(x='Date',y='Recovered',data=covid_data_date, color='green')


# In[ ]:


#Total cases and their relationship
combined=pd.melt(covid_data_date,id_vars=['Date'])
fig=px.line(combined, x='Date', y='value', color='variable' )
fig.update_layout(yaxis_title='Cases', title_text='Total Cases')
fig.show()


# With new guidelines and safety measures the CDC and WHO are asking us to take, the phrase 'Social Distancing' has become a trendy term. What it essentially means is that we need to avoid being in a crowd of more than 10 and also stay atleast 3 ft apart from each other while making conversation with anyone. This practice along with the other safety measures we are being asked to take should flatten the curve on this graph in the near future.

# In[ ]:


#Total no.of cases and their outcomes
total_cases=covid_data_date[(covid_data_date.Date==max(covid_data_date['Date']))]
print('Total Confirmed cases till date are', sum(total_cases['Confirmed']))
print('Total Death cases till date are', sum(total_cases['Deaths']))
print('Total Recoverd cases till date are', sum(total_cases['Recovered']))


# In[ ]:


covid_data_ctry = covid_data[covid_data["Date"]==covid_data["Date"].max()].groupby(["Country"])[["Confirmed", "Recovered", "Deaths"]].sum().reset_index()
#covid_data_ctry.sort_values(['Confirmed','Deaths','Recovered'],ascending=[False,False,False])
fig=px.treemap(covid_data_ctry, path=['Country'], values='Confirmed', title='Confirmed by Country')
fig.show()
#fig = px.choropleth(covid_data_ctry['Country']!='China'], locations="iso_alpha",color="Confirmed", # lifeExp is a column of gapminder hover_name="Country", # column to add to hover information color_continuous_scale=px.colors.sequential.Electric)
#fig.show()


# On March 26th, we saw that the US surpassed both Italy and China in their no.of cases to be on the first position. 

# In[ ]:


plt.figure(figsize=(15,15))
plt.xlabel('Cases', fontsize=18)
plt.ylabel('Country', fontsize=18)
plt.xticks(rotation=90)
country_confirmed=covid_data_ctry[covid_data_ctry["Confirmed"]>100].sort_values(["Confirmed"],ascending=False).head(25)
Country1=sns.barplot(x='Confirmed',y='Country',data=country_confirmed, color='blue')


# As of MArch 26th, the top 3 countries- US, China and Italy are neck to neck on the total no.of confirmed cases. However, the no. of cases added everyday in China have significantly decreased.

# In[ ]:


plt.figure(figsize=(15,15))
plt.xlabel('Cases', fontsize=18)
plt.ylabel('Country', fontsize=18)
plt.xticks(rotation=90)
country_deaths=covid_data_ctry[covid_data_ctry["Deaths"]>10].sort_values(["Deaths"],ascending=False).head(25)
Country2=sns.barplot(x='Deaths',y='Country',data=country_deaths, color='red')


# As of March 26th, in No. of Deaths, Italy and Spain have surpassed China. Both of these countries are overwhelmed by the influx of new cases. They do not have enough beds to support in the hospitals.

# In[ ]:




