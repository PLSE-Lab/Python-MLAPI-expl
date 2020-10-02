#!/usr/bin/env python
# coding: utf-8

# # *Data Interpretation COVID19 India*

# ![](https://www.railway-technology.com/wp-content/uploads/sites/24/2020/03/connection-4884862_1280.jpg)

# # *First of all, Apart from notbook and analysis.I want to say that please don't take this virus lightly.It has already affected more than a million people. 
# # Strictly abide to the decisions of the government and stay within your houses during lockdown period.*

# # IMPORT MATERIAL 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.subplots import make_subplots
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/covid192/covid19 - covid19.csv")
coordinates = pd.read_csv("../input/coordi/Indian Coordinates.csv")
data.describe(include="all")


# In[ ]:


data.head()


# In[ ]:


print(pd.isnull(data).sum())


# In[ ]:


Recovered = data[data['Current Status'] == "Recovered"]
Recovered_count = Recovered['1'].count()


# In[ ]:


Active = data[data['Current Status'] == "Hospitalized"]
Active_count = Active['1'].count()
Death = data[data['Current Status'] == "Deceased"]
Death_count = Death['1'].count()
print('There are', Recovered_count, 'recovered',Death_count,'deaths and',Active_count,'active cases.' )


#  **As you guys could see, Most of the Indians who were tested positive are in between 20-30.**

# In[ ]:


male = data[data.Gender=='M']
female = data[data.Gender=='F']


# In[ ]:


plt.figure(figsize=(15, 5))
plt.title('Gender')
data.Gender.value_counts().plot.bar();
print("Male = M and Female = F")


# In[ ]:


df = pd.DataFrame(data) 
df['Recovered'] = data['Current Status']
df['Death'] = data['Current Status']
df['Active'] = data['Current Status']
data.head()


# In[ ]:


embarked_mapping = {"Recovered": 1, "Hospitalized": 0, "Deceased": 0}
data['Recovered'] = data['Recovered'].map(embarked_mapping)
active_mapping = {"Recovered": 0, "Hospitalized": 1, "Deceased": 0}
data['Active'] = data['Active'].map(active_mapping)
death_mapping = {"Recovered": 0, "Hospitalized": 0, "Deceased": 1}
data['Death'] = data['Death'].map(death_mapping)
data


# In[ ]:


plt.figure(figsize=(25,10))
df=pd.DataFrame({'Nationality':data['Nationality'].value_counts().index,'Count':data['Nationality'].value_counts().values})
ax=sns.barplot(x="Count",y="Nationality",data=df, palette=sns.dark_palette("blue", reverse=True))
for i in ax.patches:
    ax.text(i.get_width()+0.50, i.get_y()+0.50,             str(int(i.get_width())), fontsize=15,color='black')
ax.set_title('Count of people affected and their nationality', pad=20)
plt.xlabel('Count of people affected', fontsize=20)
plt.ylabel('Nationality', fontsize=20)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=14)
plt.show()


# # Most of the affected people are Indians only.
# 

# In[ ]:


plt.figure(figsize=(20,5))
df=pd.DataFrame({'Type of transmission':data['Type of transmission'].value_counts().index,'Count':data['Type of transmission'].value_counts().values})
ax=sns.barplot(x="Count",y="Type of transmission",data=df, palette=sns.dark_palette("blue", reverse=True))
for i in ax.patches:
    ax.text(i.get_width()+0.50, i.get_y()+0.50,             str(int(i.get_width())), fontsize=15,color='black')
ax.set_title('Count of people affected and their nationality', pad=20)
plt.xlabel('Count of people affected', fontsize=20)
plt.ylabel('Type of transmission', fontsize=20)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=8)
plt.show()


# # Reason for most of the covid19 cases is still unknown.

# In[ ]:


plt.figure(figsize=(25,20))
ax=sns.barplot(data['Detected State'].value_counts().values,data['Detected State'].value_counts().index, palette=sns.dark_palette("blue", reverse=True))
for i in ax.patches:
    ax.text(i.get_width()+0.50, i.get_y()+0.50,             str(int(i.get_width())), fontsize=12,color='black')
ax.set_title('Count of detected cases in different states of India.', pad=20)
plt.xlabel('Count of people detected', fontsize=20)
plt.ylabel('State', fontsize=20)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=15)
plt.show()


# In[ ]:


data['Date Announced'] = data['Date Announced'].astype('datetime64[ns]') 

data['Date Announced'] = data['Date Announced'].dt.strftime('%m/%d/%Y')
data


# In[ ]:


recent_stats = data.groupby('Date Announced', as_index = False)['Active', 'Death', 'Recovered'].sum()
sorted_stats = recent_stats.sort_values(by = 'Date Announced', ascending = False)
sorted_stats.head()


# In[ ]:


fig = go.Figure() 
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%d-%m-%y')
ax.xaxis.set_major_formatter(myFmt)
fig.add_trace(go.Scatter(
                x= sorted_stats['Date Announced'],
                y= sorted_stats['Active'],
                name = "Active Cases",
                line_color= "deepskyblue",
                opacity= 0.8))
fig.update_layout(title_text= "Daily reported covid cases across India ")

fig.show()


# In[ ]:


fig2 = go.Figure() 

fig2.add_trace(go.Scatter(
                x= recent_stats['Date Announced'],
                y= recent_stats['Death'],
                name= "Total Deaths",
                line_color= "gray",
                opacity= 0.8))
fig2.add_trace(go.Scatter(
                x= recent_stats['Date Announced'],
                y= recent_stats['Recovered'],
                name= "Recovered",
                line_color= "deeppink",
                opacity= 0.8))

fig2.update_layout(title_text= "Death and recovered cases across India on different DATES.")
fig2.show()


# In[ ]:


plt.figure(figsize=(25,15))
ax=sns.barplot(data['Detected District'].value_counts()[:20].values,data['Detected District'].value_counts()[:20].index, palette=sns.dark_palette("blue", reverse=True))
for i in ax.patches:
    ax.text(i.get_width()+0.50, i.get_y()+0.50,             str(int(i.get_width())), fontsize=30,color='black')
ax.set_title('Top 20 districts with highest count of affected people.', pad=20,fontsize=40)
plt.xlabel('Count of people affected', fontsize=30)
plt.ylabel('District', fontsize=30)
# plt.rc('xtick',labelsize=0)
# plt.rc('ytick',labelsize=20)
plt.show()


# # Mumbai is the leading city with over 200 cases.

# In[ ]:


# get recent stats
age_stats = data.groupby('Age Bracket', as_index = False)['Active', 'Death', 'Recovered'].sum()
sortedage_stats = age_stats.sort_values(by = 'Age Bracket', ascending = False)
sortedage_stats.head()


# In[ ]:


fig4 = go.Figure()
fig4.add_trace(go.Scatter(
                x= sortedage_stats['Age Bracket'],
                y= sortedage_stats['Active'],
                name = "Active Cases",
                line_color= "deepskyblue",
                opacity= 0.8))
fig4.update_layout(title_text= "Daily reported covid cases with different AGE GROUP ")

fig4.show()


# # Most affected age group with covid19 is 20 to 35 years old.

# In[ ]:


fig3 = go.Figure() 

fig3.add_trace(go.Scatter(
                x= age_stats['Age Bracket'],
                y= age_stats['Death'],
                name= "Total Deaths",
                line_color= "gray",
                opacity= 0.8))
fig3.add_trace(go.Scatter(
                x= age_stats['Age Bracket'],
                y= age_stats['Recovered'],
                name= "Recovered",
                line_color= "deeppink",
                opacity= 0.8))

fig3.update_layout(title_text= "Death and recovered cases with different AGE GROUP")
fig3.show()


# # Most of the Deaths are occuring in between age group 60-70.

# In[ ]:



f, ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="Age Bracket", y="Active", data=data,
             color="black",label = "Active")
ax=sns.scatterplot(x="Age Bracket", y="Recovered", data=data,
             color="red",label = "Recovered")
ax=sns.scatterplot(x="Age Bracket", y="Death", data=data,
             color="blue",label = "Death")
plt.plot(data['Age Bracket'],data.Active,zorder=1,color="black")
plt.plot(data['Age Bracket'],data.Recovered,zorder=1,color="red")
plt.plot(data['Age Bracket'],data.Death,zorder=1,color="blue")


# In[ ]:


plt.figure(figsize=(25,10))
ax=sns.swarmplot(data['Current Status'],data['Age Bracket'],palette=sns.dark_palette("blue", reverse=True))
ax.set_title('Graph between age and current status of affected people', pad=20,fontsize=20)
plt.xlabel('Current Status', fontsize=20)
plt.ylabel('Age Bracket', fontsize=20)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)


# # It was just the beginning, I will update the data and include more Data Visuals daily.

# # Ways to prevent Corona:
# 
# * Wash your hands regularly for 20 seconds, with soap and water or alcohol-based hand rub
# * Cover your nose and mouth with a disposable tissue or flexed elbow when you cough or sneeze
# * Avoid close contact (1 meter or 3 feet) with people who are unwell
# * Stay home and self-isolate from others in the household if you feel unwell
# * Don't touch your eyes, nose, or mouth if your hands are not clean

# # Thank you. 
