#!/usr/bin/env python
# coding: utf-8

# # Corona Virus 2019
# This Notebook have visualize info for Corona Virus spread all over the word also some insights from the data like which country has maximum spread and which country has max success rate in coronavirus cure.
# 

# # Insight
# Sri Lanka has max(more than 45% people are recoverd) recoverd ratio.
# 

# In[ ]:


import pandas as pd
data = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")


# In[ ]:


data.info()


# # Data Cleaning and EDA

# In[ ]:


data['Date'] = pd.to_datetime(data.Date)
data['Date'] = pd.DatetimeIndex(data['Date']).date
data['Date'] = data.Date.apply(str)


# In[ ]:


data.groupby('Date').sum()['Confirmed']


# In[ ]:


c = data.groupby('Country').sum()
c = c.drop(['Sno'],axis = 1)
c.style.background_gradient(cmap='rainbow')


# In[ ]:


print("Total number of Corona Virus Confirmed Case are " + str(sum(data.Confirmed)))
print("Total number of Corona Virus Deaths are " + str(sum(data.Deaths)))
print("Total number of Corona Virus Recovered " + str(sum(data.Recovered)))


# ## Visualization

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size


# In[ ]:


groupedvalues = data.groupby('Date').sum().reset_index()
sns.set(style="whitegrid")
sns.barplot(x='Confirmed',y='Date',data=groupedvalues)
plt.title('Number of Confirmed Corona Virus cases each day from 22-01-2020 to 18-02-2020')


# In[ ]:


sns.barplot(x='Deaths',y='Date',data=groupedvalues)
plt.title('Number of Deaths due to Corona Virus from 22-01-2020 to 18-02-2020')


# In[ ]:


sns.barplot(x='Recovered',y='Date',data=groupedvalues)
plt.title('Number of Recovered cases from 22-01-2020 to 18-02-2020')


# In[ ]:


groupedvalues = groupedvalues.drop(['Sno'], axis=1)


# In[ ]:


df = groupedvalues.melt(id_vars=["Date"], 
        var_name="Type", 
        value_name='NumberOfPeople')


# In[ ]:


fig, ax = plt.subplots()
sns.set(style="whitegrid")
sns.barplot(x='Date',y='NumberOfPeople',data=df,hue = 'Type')
fig.autofmt_xdate()


# ## Corona Virus Spread On World Map

# In[ ]:


import plotly.express as px


# In[ ]:


groupedvalues = data.groupby('Country').sum().reset_index()
groupedvalues = groupedvalues.drop(['Sno'], axis=1)


# In[ ]:


fig = px.scatter_geo(groupedvalues, locations="Country", locationmode='country names', 
                     color="Confirmed", hover_name="Country", range_color= [0, 20], projection="natural earth",
                    title='Spread across the world')
fig.update(layout_coloraxis_showscale=False)
fig.show()


# # Ratio of Recovered people in each country

# In[ ]:


groupedvalues['RecoveredRatio'] = (groupedvalues['Recovered'] * 100) / groupedvalues['Confirmed']


# In[ ]:


fig, ax = plt.subplots()
sns.set(style="whitegrid")
sns.barplot(x='Country',y='RecoveredRatio',data=groupedvalues)
fig.autofmt_xdate()


# # Insight
# Sri Lanka has max(more than 45% people are recoverd) recoverd ratio.

# In[ ]:




