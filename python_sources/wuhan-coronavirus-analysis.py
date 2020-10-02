#!/usr/bin/env python
# coding: utf-8

# ## Usual Libraries

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualisation libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('darkgrid')
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5

# Disable warnings 
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# According to World Health Organization (WHO), "Coronaviruses (CoV) are a large family of viruses that cause illness ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS-CoV) and Severe Acute Respiratory Syndrome (SARS-CoV). A novel coronavirus (nCoV) is a new strain that has not been previously identified in humans." 
# 
# It has been more than one month since the first case of this new zootonic virus has been reported. Wuhan, the capital of Hubei province of China is the epicenter of this new virus. The situation reports about this outbreak can be found [here.](https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports/)
# 
# In this notebook I will try to get a one month overview of the outbreak. 

# ## Importing Dataset

# In[ ]:


df= pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv",)
df.head()


# In[ ]:


# Let's check the infos
df.info()


# In[ ]:


# Convert Last Update column to datetime64 format
df['Last Update'] = df['Last Update'].apply(pd.to_datetime)
df.drop(['Sno'],axis=1,inplace=True)
df.head()


# We know this kind of viruses can spread primarily through contact with an infected person, through respiratory droplets generated when a person, for example, coughs or sneezes, or through droplets of saliva or discharge from the nose. The same happened with 2019-nCov, so let's see in which countries are mostly affected by this virus.

# In[ ]:


countries = df['Country'].unique().tolist()
print(countries)

print("\nTotal countries affected by virus: ",len(countries))


# ## Scenerio till 31 January, 2020

# In[ ]:


from datetime import date
latest_data = df[df['Last Update'] > pd.Timestamp(date(2020,1,31))]

latest_data.head()


# In[ ]:


print('Globally Confirmed Cases: ',latest_data['Confirmed'].sum())
print('Global Deaths: ',latest_data['Deaths'].sum())
print('Globally Recovered Cases: ',latest_data['Recovered'].sum())


# In[ ]:


# Let's look the various Provinces/States affected

latest_data.groupby(['Country','Province/State']).sum()


# In[ ]:


# Creating a dataframe with total no of cases for every country


cases = pd.DataFrame(latest_data.groupby('Country')['Confirmed', 'Deaths', 'Recovered'].sum())
cases['Country'] = cases.index
cases.index=np.arange(1,28)

global_cases = cases[['Country','Confirmed','Deaths', 'Recovered']]
global_cases


# In[ ]:


map_data = pd.DataFrame({
   'name':list(global_cases['Country']),
   'lat':[-25.27,12.57,56.13,61.92,46.23,51.17,22.32,20.59,41.87,36.2,22.2,35.86,4.21,28.39,12.87,61.52,1.35,35.91,40.46,7.87,60.12,23.7,15.87,55.37,37.09,23.42,14.06,],
   'lon':[133.78,104.99,-106.35,25.75,2.21,10.45,114.17,78.96,12.56,138.25,113.54,104.19,101.98,84.12,121.77,105.31,103.82,127.77,3.74,80.77,18.64,120.96,100.99,3.43,-95.71,53.84,108.28],
})
fig= go.Figure()
fig.add_trace(go.Scattergeo(
        lat= map_data['lat'],
        lon= map_data['lon'],
        mode= 'markers',
        marker= dict(
            size= 12,
            color='rgb(255, 0, 0)',
            opacity= 0.7
        ),
        text= map_data['name'],
        hoverinfo= 'text'
    ))

fig.add_trace(go.Scattergeo(
        lat= map_data['lat'],
        lon= map_data['lon'],
        mode= 'markers',
        marker= dict(
            size= 8,
            color= 'rgb(242, 177, 172)',
            opacity= 0.7
        ),
        hoverinfo= 'none'
    ))

fig.update_layout(
        autosize= True,
        hovermode= 'closest',
        showlegend= False,
        title_text= 'Countries with reported confirmed cases, Deaths, Recovered of 2019-nCoV,<br>31 January, 2020',
    geo= go.layout.Geo(
        showframe= False,
        showcoastlines= True,
        showcountries= True,
        landcolor= "rgb(225, 225, 225)",
        countrycolor= "blue",
        coastlinecolor= "blue",
        projection_type= "natural earth"
    ))

 
fig.show()


# **Let's see if any countries have death cases other than China:**

# In[ ]:


global_cases.groupby('Country')[ 'Deaths'].sum()


# **So till 31 January, 2020 no other countries have reported deaths. So let's focus on the present state of China:**

# ## Focus in China

# In[ ]:


#Mainland China
China = latest_data[latest_data['Country']== 'Mainland China']
China


# Let's look at the Confirmed vs Recovered figures of Provinces of China other than Hubei

# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))


sns.barplot(x="Confirmed", y="Province/State", data=China[1:],
            label="Confirmed", color="r")


sns.barplot(x="Recovered", y="Province/State", data=China[1:],
            label="Recovered", color="g")


sns.barplot(x="Deaths", y="Province/State", data=China[1:],
            label="Deaths", color="b")

# Add a legend and informative axis label
ax.set_title('Confirmed vs Recovered vs Death figures of Provinces of China other than Hubei', fontsize=15, fontweight='bold', position=(0.63, 1.05))
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 40), ylabel="",
       xlabel="Stats")
sns.despine(left=True, bottom=True)


# **Now let's see which provinces have the highest percentage of confirmed cases:**

# In[ ]:


percentage = pd.DataFrame(China.groupby('Province/State').sum()['Confirmed']).reset_index()
fig = go.Figure(data= [go.Pie(labels= percentage['Province/State'], values= percentage.Confirmed)])
fig.update_layout(title="Confirmed cases in province/states of Mainland China")
fig.show()


# **The highest deat tolls:**

# In[ ]:


percentage = pd.DataFrame(China.groupby('Province/State').sum()['Deaths']).reset_index()
fig = go.Figure(data= [go.Pie(labels= percentage['Province/State'], values= percentage.Deaths)])
fig.update_layout(title="Death tolls in province/states of Mainland China")
fig.show()


# **Also checking the recovery rate:**

# In[ ]:


percentage = pd.DataFrame(China.groupby('Province/State').sum()['Recovered']).reset_index()
fig = go.Figure(data= [go.Pie(labels= percentage['Province/State'], values= percentage.Recovered)])
fig.update_layout(title="Recovery rates in province/states of Mainland China")
fig.show()


# ## Conclusion

# **As Wuhan is the epicenter of the coronavirus and capital of the Hubei province, so it has the highest rate in all three features. The recovery rate is still not so high. But we hope and pray this pandemic ends soon and people can start living a normal fear free life again in China and all over the world.**

# **Special thanks to [Parul Pandey](https://www.kaggle.com/parulpandey) for her amazing work on this dataset from which I got my motivation.**
