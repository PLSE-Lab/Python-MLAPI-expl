#!/usr/bin/env python
# coding: utf-8

# # Novel Corono Virus
# 
# **Backstory:** 
# The recent virus belong to a family of virus named 'Corona'. 
# It is called Novel because its new, and no name hasn't been thought of yet. 
# This class of virus usually migrates from animals to humans, thus contagious. 
# 
# **Symptoms:**
# The symtoms usually are seasonal flu, which makes it difficult to identify. 
# 
# **Threats:**
# The virus spreads really faster, which is good and bad at the same time. 
# If you are confused as I was when I first heard it, keep reading. 
# A virus which can spread easily, is much less life threathning. 
# That is why the mortality rate of Corona Virus is around 2% as per scientist across the Globe. 
# 
# **Verdict:**
# So, what's the verdict? Well, there is no reason to panic, but its defintely a matter of concern. 
# Its because, it will take more than one year to derieve any cure for the virus. 
# In the mean time, it's our immune system that will have to fight. 
# So, stay clean and eat healthy.
# 
# 
# ### The Dataset
# Based on dataset available which contains geographical and medical records, we will learn the impact of Corona Virus through exploratory data analysis with special focus on China. 
# 
# *Please Note:* I have just started on this notebook. So,please feel free to comment and provide feedback to improve the notebook. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.graph_objs as go
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## **Import Dataset and Preview**

# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
df.head()


# ### Format Dataset
# - Format Date and Time Column
# - Rename Column Name
# - Drop Sno Column

# In[ ]:


df['Date'] = df['Date'].apply(pd.to_datetime)
df['Last Update'] = df['Last Update'].apply(pd.to_datetime)
df = df.rename(columns={'Last Update':'Last_Update'})
df.drop(columns='Sno',axis=1,inplace=True)
df.head()


# ### Feature Engineering - I
# - Get Day and Month from Data-Time Column
# 
# ![](http://)Through this technique, we can analyze spread of Corono Virus with respect to time (i.e. from January to February, 2020).

# In[ ]:


df['ev_month'] = [df.Date[i].month for i in range(df.shape[0])]
df['ev_day']   = [df.Date[i].day   for i in range(df.shape[0])]
df.drop(columns='Date',axis=1,inplace=True)

df['ls_month'] = [df.Last_Update[i].month for i in range(df.shape[0])]
df['ls_day']   = [df.Last_Update[i].day   for i in range(df.shape[0])]
df.drop(columns='Last_Update',axis=1,inplace=True)

df.head()


# ### Get an Indepth Idea on Dataset

# In[ ]:


df.describe(include='all')


# ### Feature Engineering - II
# ![](http://)Mainland China and China are same, thus renaming the entries.

# In[ ]:


df.Country[df.Country=='Mainland China']='China'
df.groupby('Country').sum()[['Confirmed','Deaths','Recovered']]


# ## Impact of Novel Corona Virus (Outside China)
# 
# > China has the highest number of records of Corona Virus infection confirmation, when compared to rest of the world. Therefore, the numbers obtained for China is out of scale from rest of the World, as of now. Thus, special attention is given on China in subsequent analysis, while currently focussing on rest of the world.

# In[ ]:


df_wc = df[df.Country!='China']
g= df_wc.groupby('Country').sum()[['Confirmed','Deaths','Recovered']]

fig = make_subplots(rows=3, cols=1,subplot_titles=("Confirmed", "Deaths", "Recovered"))
fig.add_trace(go.Bar(x=g.index, y=g.Confirmed),row=1, col=1)
fig.add_trace(go.Bar(x=g.index, y=g.Deaths   ),row=2, col=1)
fig.add_trace(go.Bar(x=g.index, y=g.Recovered),row=3, col=1)
fig.update_layout(height=700, width=1000, title_text="Corona Virus Report (Except China)")
fig.show()


# The mortality rate for most nations is zero, except for Hong Kong & Phillipines. However, the mortality rate is quite low (less than 4) for these nations. It is interesting to observe that Thailand have the highest number of recovery followed by Australia, Japan and Vietnam. 
# 
# Citizen from Brazil, Belgium, Mexico have reported zero infection from Corona Virus. Either, no one recently have visited China or came in contact with any infected person. This, implies, with restricted travel allowances, Corona Virus can be reasonably quanrantined. 

# ## Impact of Novel Corona Virus (Within China)
# 
# By focussing on China, we can have a much granular view i.e. state-wise assesment of the infection. This will expedite the search for origin of virus. 

# In[ ]:


g = df[df.Country=='China'].groupby('Province/State').sum()[['Confirmed','Deaths','Recovered']]
fig = make_subplots(rows=3, cols=1,subplot_titles=("Confirmed", "Deaths", "Recovered"))
fig.add_trace(go.Bar(x=g.index, y=g.Confirmed),row=1, col=1)
fig.add_trace(go.Bar(x=g.index, y=g.Deaths   ),row=2, col=1)
fig.add_trace(go.Bar(x=g.index, y=g.Recovered),row=3, col=1)
fig.update_layout(height=800, width=1000, title_text="Corona Virus Report (In States of China)")
fig.show()


# From previous observation, it can be observed that Hubel reports maximum number of infection compared to rest of the Country. The death and recovery is also relatively high for Hubel, given it has the highest number of confirmation on infection. However, number of deaths or recovery seems to be very less, i.e. 1/40 times of number of confirmed infection. 
# 
# Although, Hubel is considered as a top priority, other states must be taken into consideration given the virus is contagious. This means that all states must be studied further, for number of confirmed infection, deaths and recovery. 

# In[ ]:


g = g[g.Confirmed<max(g.Confirmed)]
fig = make_subplots(rows=3, cols=1,subplot_titles=("Confirmed", "Deaths", "Recovered"))
fig.add_trace(go.Bar(x=g.index, y=g.Confirmed),row=1, col=1)
fig.add_trace(go.Bar(x=g.index, y=g.Deaths   ),row=2, col=1)
fig.add_trace(go.Bar(x=g.index, y=g.Recovered),row=3, col=1)
fig.update_layout(height=700, width=1000, title_text="Corona Virus Report (In States of China)")
fig.show()


# Most Chinese states have very low death rates(less than 20 or 1/200 times of confirmed infection), but moderate recovery rates (closer to 200 or 1/20  times of confirmed infection). This implies, despite increasing number of infections, recovery is still possible. However, much higher recovery rate should be attained. 

# ## Impact of Corona Virus on Other Nations
# 
# Besides China, other nations which have high number of confirmed infection are Australia, Hong Kong, Japan, Macau, Malaysia, Singapore, South Korea, Taiwan, Thailand, and US. Out of all countries, data at state level is available for Australia and US only, which is analyzed below. 

# In[ ]:


print("Granular view for following nations were available\n")
g4 = df[df.Country=='Australia'].groupby('Province/State').sum()[['Confirmed','Deaths','Recovered']]
print("\nStats for Australia\n",'_'*50,'\n',g4)
g4 = df[df.Country=='US'].groupby('Province/State').sum()[['Confirmed','Deaths','Recovered']]
print("\nStats for United States of America\n",'_'*50,'\n',g4)


# No death due to Corona virus is reported in US and Australia, which is really a good news. 
# 
# However, number of confirmed infection is higher in Australia when compared to US. 
# Especially, New South Wales and Victoria needs special attention to bring down the infection rate.

# ## Growth of Corona Virus across Time in China

# In[ ]:


dft = df[df.Country=='China']
g1 = pd.DataFrame(dft[['Country','ev_day','ev_month','Confirmed']].groupby(['ev_month','ev_day']).sum()['Confirmed'])

a=[i for i in range(g1.shape[0])]

fig = px.bar(x=a, y=g1.Confirmed)
fig.update_layout(height=300, width=800, title_text="Corona Virus (In China)")

fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = [i for i in range(g1.shape[0]+1)],
        ticktext = g1.index
    )
)
fig.show()


# The graph shows data of infections due to Corona Virus from 22/01/2020 to 04/02/2020 in China. 
# The trend seems to be increasing with time, at an alarming rate i.e from 549 to 24290 in just two weeks. 
# Thus, it is imperative to take action immediately to quarantine further spread of this Virus and find a cure ASAP.

# ## Growth of Corona Virus across Time outside China

# In[ ]:


dft = df[df.Country!='China']
g2 = pd.DataFrame(dft[['Country','ev_day','ev_month','Confirmed']].groupby(['ev_month','ev_day']).sum()['Confirmed'])

a=[i for i in range(g1.shape[0])]

fig = px.bar(x=a, y=g2.Confirmed)
fig.update_layout(height=300, width=800, title_text="Corona Virus (Rest of the World)")
fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = [i for i in range(g1.shape[0]+1)],
        ticktext = g1.index
    )
)
fig.show()


# The graph shows data of infections due to Corona Virus from 22/01/2020 to 04/02/2020 outside of China. 
# The trend seems to be increasing with time, at a much slower pace(compared to China) i.e from 6 to 213 in two weeks. 
# Though, the rate of spread is quite low, it is still imperative to take action immediately to quarantine further spread of this Virus and find a cure ASAP. 

# ### Key Points:
# 
# - China have highest number of infected people & deaths, especially a state 'Hubel'.
# - Thailand, Australia, Japan and Vietnam have the highest recovery rate.
# - The spread of Corona Virus is spreading at an alarming rate in China, which needs to be mitigated ASAP.
# - Restricted travel to and fro from China, or keeping distance from people who are already infected can act as a quarantine measure. 

# In[ ]:




