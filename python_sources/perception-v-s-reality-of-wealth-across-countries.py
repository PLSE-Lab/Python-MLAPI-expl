#!/usr/bin/env python
# coding: utf-8

# **Analysis of country wealth by per capita GDP (PPP $)**
# 
# <img src="https://s7.gifyu.com/images/ppp-usd.gif">

# If you ask an average educated person on street in India about their perception of how good India is doing as an economy, you will get to hear India is now **5th world largest **economy by GDP (nominal), **so we are doing good**.
# 
# Perception is we might be behind some developed country and ofcourse China, but we are better than lot other like our neighbors (Srilanka, Pakistan), african countries (Nigeria) and definitely better than asian countries like Phillipines and Vietnam. **But is this really true?** Lets find it out by simple exploratory data analysis. 
# 
# 

# In[ ]:


# As first step, I have imported all required libraries.


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import os


# Then loaded and read data from World Bank (https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.KD)

# In[ ]:


print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/API_NY.GDP.PCAP.PP.CD_DS2_en_csv_v2_1068882.csv', skiprows=(0,3), header = 1)


# In[ ]:


df.head()


# In[ ]:


col_to_drop= df.columns[4:33]  # There are multiple columns with no data which, hence deleted. 


# In[ ]:


df = df.drop(col_to_drop, axis=1)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df = df.dropna(axis=1,thresh=3)   # we have consistent data from 1990 onwards. Still there are some countries 
# where data is missing. Setting threshold of 3 null values. FOr all rows with 3 or more null values, row will be deleted


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df['Country Name'].unique()   # Fdinding out list of countries for which data is present


# List of interested countries chosen

# In[ ]:


a=df['Country Name'].isin(['India','Nigeria','Bangladesh','Philippines', 'Pakistan','Sri Lanka', 'Vietnam'])


# In[ ]:


df= df[a]


# In[ ]:


df.shape


# In[ ]:


df.head()  # details of only chosen countries


# In[ ]:


col = df.columns[1:4]  # Column 1 to 4 look rdundant, hence removing


# In[ ]:


df = df.drop(col, axis=1)


# In[ ]:


df.head()


# Will need to transpose the data to analyse and plot graphs

# In[ ]:


df_T= df.transpose(copy=True)


# In[ ]:


df_T.shape


# In[ ]:


df_T


# In[ ]:


df_T.index.name = 'Year'  # Setting year as index


# In[ ]:


new_header = df_T.iloc[0] #grab the first row for the header


# In[ ]:


df_T = df_T[1:] #take the data less the header row


# In[ ]:


df_T.columns = new_header #set the header row as the df header


# In[ ]:


df_T.index


# In[ ]:


df_T.columns  # columns in new datafram


# In[ ]:


df_T.index= pd.to_datetime(df_T.index,format='%Y').year  # Setting index year in to date format


# In[ ]:


type(df_T.index)  # checking data type


# In[ ]:


## Draw a simple line plot to see movement of Per Capita GDP ($) 


# In[ ]:


fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(df_T.index, df_T['Bangladesh'], label= 'Bangladesh')
ax.plot(df_T.index, df_T['Sri Lanka'], label= 'Sri Lanka')
ax.plot(df_T.index, df_T['India'], label= 'India')
ax.plot(df_T.index, df_T['Nigeria'], label= 'Nigeria')
ax.plot(df_T.index, df_T['Vietnam'], label= 'Vietnam')
ax.plot(df_T.index, df_T['Pakistan'], label= 'Pakistan')
ax.set_xlabel('Years')
ax.set_ylabel('GDP Per Capita PPP')
ax.set_title('GDP per capita PPP over the years')
plt.legend(loc="upper left")


# 

# In[ ]:


s=df_T.loc[1990]  # Slicing a series to plot a horizontal bar chart


# Draw a single horizontal barc hart for year 1990 for selected counties

# In[ ]:


fig, ax = plt.subplots(figsize=(4, 2.5), dpi=144)
colors = plt.cm.Dark2(range(6))
y = s.index
width = s.values
ax.barh(y=y, width=width, color=colors);
ax.set_xlabel('GDP PPP per Capita ($)')


# In[ ]:


# Draw barc chart for 3 years


# In[ ]:


fig, ax_array = plt.subplots(nrows=1, ncols=3, figsize=(7, 2.5), dpi=144, tight_layout=True)
dates = [1990,1991,1992]
for ax, date in zip(ax_array, dates):
    s = df_T.loc[date].sort_values()
    ax.barh(y=s.index, width=s.values, color=colors)
    ax.set_title(date, fontsize='smaller')
    ax.set_xlabel('GDP PPP Per Capita $')


# In[ ]:


# Rank individual countries based on PPP that year


# In[ ]:


df_T.loc[1990].rank()


# In[ ]:


df_T_rank = df_T.rank(axis=1)
df_T_rank


# Draw barchart based on ranking each year

# In[ ]:


fig, ax_array = plt.subplots(nrows=1, ncols=3, figsize=(7, 2.5), dpi=144, tight_layout=True)
dates = [1990,1991,1992]
for ax, date in zip(ax_array, dates):
    s = df_T.loc[date]
    y = df_T.loc[date].rank().values
    ax.barh(y=y, width=s.values, color=colors,tick_label=s.index)
    ax.set_title(date, fontsize='smaller')
    ax.set_xlabel('GDP PPP Per Capita $')


# In[ ]:


labels = df_T.columns


# Finally, write code for Running Plot by using codes defined above. (Credit- https://medium.com/dunder-data/create-a-bar-chart-race-animation-in-python-with-matplotlib-477ed1590096, https://www.kaggle.com/parulpandey/who-earns-the-most-in-sports/data) 

# In[ ]:


from matplotlib.animation import FuncAnimation

def init():
    ax.clear()
    ax.set_ylim(1, 7.5)

def update(i):
    for bar in ax.containers:
        bar.remove()
    y = df_T_rank.iloc[i]
    width = df_T.iloc[i]
    ax.barh(y=y, width=width, color=colors, tick_label=labels)
    date_str = df_T.index[i]
    ax.set_title(f'PPP GDP - {date_str}', fontsize='smaller')
    ax.set_xlabel('GDP Per Capita (PPP $) over years')
    
fig = plt.Figure(figsize=(8, 4), dpi=130)
ax = fig.add_subplot()
anim = FuncAnimation(fig=fig, func=update, init_func=init, frames=len(df_T), 
                     interval=500, repeat=False)


# In[ ]:


from IPython.display import HTML
html = anim.to_jshtml()
HTML(html)


# **We can see that per capita income of many countries like Srilanka, Vietnam,Phillipines is better than India's. Also in past, not long back per capita incomes of Pakistan and Nigeria too were better than of India.**
