#!/usr/bin/env python
# coding: utf-8

#  # Terror in Israel from 1971 - 2016
# In this notebook I want to give an overview about terror attacks in Israel. The following results are based on the provided **Global Terrorism Database**. Since the establishment of the Israeli state in 1948 its citizens faced a large amount of terrorist attacks. Even if I can't guarantee for 100% completeness of the datasource, this data exploration should at least give an overview about the victims, attackers and development of terrorism in Israel since the last 45 years. 
# 
# *Disclaimer: This notebook is work in progress. I'm happy to receive feedback and if you spot any mistakes in my code. I'm only visualizing data here, so keep political discussions out of the comment section please. Thank you. As stated, the dataset seems to be not 100% complete, so the number of victims shown here might be lower as in reality.*
# 
# **Example data:**

# In[ ]:


import numpy as np
import pandas as pd
import folium
import warnings
warnings.filterwarnings('ignore')
import datetime
import calendar

import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"

from IPython.display import display, Markdown, Latex
matplotlib.rcParams['figure.figsize'] = (12.0, 5.0)

df = pd.read_csv("../input/globalterrorismdb_0617dist.csv" ,encoding='ISO-8859-1')
df_il = df[df.country_txt == "Israel"]
df_il.head()


# ## Number of terror attacks over time
# Sadly, there wasn't a single period without terrorism in the last 45 years in Israel. Tensions between Israel and its neighbouring states/ areas got stronger after the **Six Days War** in 1968 and as seen in the data, are present until today. The number of fatalities is usually in the same range with the number of attacks, but there are two peeks visible: The first around 2002 is related to the rise of violence after the start of the Second Intifada in 2000. The rise of attacks at the end of the curve seems to be related to the so called **Wave of Terror** that started in September 2015 and lasted into the first half of 2016. It's also called **Knife Intifada**, because a large amount of the attacks were stabbing attacks by Palestinians against Israeli citizens. 

# In[ ]:


data = df_il.groupby("iyear").nkill.sum()
data = data.reset_index()
data.columns = ["Year", "Number of fatalities"]

ax = data.plot(x="Year", y="Number of fatalities", legend=False)
ax2 = ax.twinx()

data = df_il.groupby("iyear").nkill.count()
data = data.reset_index()
data.columns = ["Year", "Number of attacks"]

data.plot(x="Year", y="Number of attacks", ax=ax2, legend=False, color="r")
ax.figure.legend(bbox_to_anchor=(0.05, 0.92), loc="upper left")
plt.tight_layout()
plt.show()


# ## Victim groups
# The largest victim group of terror in Israel are private citizens as shown in the upcoming chart (based in the number of attacks). By looking at the other categories, I would also consider for example "Transportation", "Business" and "Educational Institution" as target groups that consist mainly of private citizens. Still, even looking at the data as is, it's clear that the primary target of the terror attacks are private citizens.

# In[ ]:


data = df_il.groupby("targtype1_txt").eventid.count().sort_values(ascending=False)[:10]
data = data.reset_index()
data.columns = ["Target", "Number of attacks"]
sns.barplot(data=data, x=data.columns[1], y=data.columns[0]);


# ## Geographical exploration over time
# The terror in Israel changed over time. In the upcoming charts, I want to give you an overview of the different decades. As you can see, depending on the years and circumstances, different terrorist groups were involved. Also the weapon types changed as stated before - this gets especially visibile during the time of the Knife Intifada in the early 2000s. One sad fact remains constant over time: The primary target group of terror in Israel is its citizens. Saturday was always the least prominent day for terror, since most likely the potential amount of damage during this quiet day is the smallest.
# 
# **Explanation for geographical visualization:** In the maps created with folium, each circle represents a terrorist attack. The radius of the circle is depending on the amount of fatalities - bigger circle means more people who got killed in the attack. <span style="color:red">Red</span> circles are terrorist attacks, where I assume based on the dataset that civilian casualities were specifically planned or at least tolerated. This includes every entry in the dataset with **none** of those target types: 
# 
# - Food or Water Supply
# - Government (Diplomatic)
# - Government (General)
# - Journalists & Media
# - Other
# - Police
# - Telecommunication
# - Terrorists/Non-State Militia
# - Utilities
# - Violent Political Party
# 
# Of course with most of those target types, terrorists also aim at or accept civilian casualities. But for visualization purpose I marked them in <span style="color:blue">blue</span> and group them as institutional target types. 
# 
# *Disclaimer: In the dataset the terrorist organization is often marked as 'Unknown'. I exclude those dataset entries in the upcoming analysis of top terrorist groups during a specific time*

# In[ ]:


for year in [[df_il.iyear.min(), 1980], [1980, 1990],
             [1990, 2000], [2000, 2010], [2010, df_il.iyear.max()]
            ]:
    
    m = folium.Map(
    location=[32.109333, 34.855499],
    zoom_start=7,
    tiles='Stamen Toner'
    )
    
    data = df_il.query("{} < iyear <= {}".format(year[0], year[1]))
    data = data.drop(data[data.iday < 1].index)
    data['weekday'] = [calendar.day_name[datetime.datetime(day.iyear, day.imonth, day.iday).weekday()] for i, day in data.iterrows()]
    data['date'] = [datetime.datetime(day.iyear, day.imonth, day.iday) for i, day in data.iterrows()]

    non_civ_target = ['Food or Water Supply', 'Government (Diplomatic)',
       'Government (General)', 'Journalists & Media', 'Other', 'Police', 'Telecommunication',
       'Terrorists/Non-State Militia', 'Utilities', 'Violent Political Party']
    
    for i in range(len(data)):
        row = data.iloc[i]
        
        if row.targtype1_txt in non_civ_target:
            color = '#6b9cff'
        elif row.targtype1_txt == 'Unknown':
            color = "#e3b57e"
        else:
            color = '#9b5353'       
        
        desc = "Type: {}; Number fatalities: {}; Number wounded: {}; Year: {}".format(row.attacktype1_txt, row.nkill, row.nwound, row.iyear)
        if not pd.isnull(row.longitude):
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=row.nkill,
                popup=desc,
                color=color,
                fill=color

            ).add_to(m)
            
    display(Markdown("<center style='background: black;'><font color='white' size='12'>Terror attacks from {} to {}</font></center>".format(year[0], year[1])))
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(15,3), facecolor='black')
    
    data_sub = data.groupby("weapsubtype1_txt").eventid.count().sort_values(ascending=False).iloc[:3]
    data_sub = data_sub.reset_index()
    data_sub.columns = ["Weapon Type", "Number of attacks"]
    sns.barplot(data=data_sub, x=data_sub.columns[1], y=data_sub.columns[0], ax=ax1)
    
    data_sub = data.groupby("targtype1_txt").eventid.count().sort_values(ascending=False).iloc[:3]
    data_sub = data_sub.reset_index()
    data_sub.columns = ["Target", "Number of attacks"]    
    sns.barplot(data=data_sub, x=data_sub.columns[1], y=data_sub.columns[0], ax=ax2)
    
    ax1.set_title('Most used weapons (top 3)')
    ax1.set_ylabel('')
    ax2.set_title('Most frequent targets (top 3)')
    ax2.set_ylabel('')   

    plt.tight_layout()
    plt.show()
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,3), facecolor='black')
    
    data_sub = data.groupby("weekday").nkill.count().sort_values(ascending=False).iloc[:]
    data_sub = data_sub.reset_index()
    data_sub.columns = ["Weekday", "Number of attacks"]    
    sns.barplot(data=data_sub, x=data_sub.columns[1], y=data_sub.columns[0], ax=ax1)


    data_sub = data.groupby("gname").eventid.count().sort_values(ascending=False)
    data_sub = data_sub.reset_index()
    data_sub = data_sub.drop(data_sub[data_sub.gname == 'Unknown'].index)[:3]

    data_sub.columns = ["Attacker", "Number of attacks"]
    sns.barplot(data=data_sub, x=data_sub.columns[1], y=data_sub.columns[0], ax=ax2)
    
    ax1.set_title('Fatalities per weekday')
    ax1.set_ylabel('')
    ax2.set_title('Most active terror groups (top 3)')
    ax2.set_ylabel('')     
    
    plt.tight_layout()
    plt.show()
    
    display(m)
    display(Markdown("<hr></hr>"))

    


# ## Overview about target and victim groups
# Which terrorist organizations are performing all of those attacks according to the data? In the next visualization you can see a heatmap, that shows for every terrorist organization how much people got killed or wounded. For visualization purpose this heatmap only shows the 10 most active terrorist groups and the 10 most targeted victim groups. The visualization shows, that Hamas is the most active group when it comes to the Transportation and Private Citizens groups.

# In[ ]:


data = df_il.groupby(["gname", "targtype1_txt"])[['nkill', 'nwound']].sum()
data = data.reset_index()
data = data[data.targtype1_txt != 'Unknown']
data = data[data.gname != 'Unknown']
data = data[data.gname.isin(list(df_il.groupby("gname").nwound.sum().sort_values(ascending=False)[:10].index.tolist()))]
data = data[data.targtype1_txt.isin(list(df_il.groupby("targtype1_txt").nwound.sum().sort_values(ascending=False)[:10].index.tolist()))]


data = data.fillna(0)
data['nvictim'] = data.nkill + data.nwound
del data['nkill']
del data['nwound']
sns.heatmap(data.pivot('gname', 'targtype1_txt', 'nvictim'),square=True, linewidths=1, linecolor='white')
plt.ylabel('')
plt.xlabel('');

