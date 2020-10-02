#!/usr/bin/env python
# coding: utf-8

# ![](https://www.beefmagazine.com/sites/beefmagazine.com/files/styles/article_featured_retina/public/0603T1-1779A-1540x800.jpg?itok=oHbTmWLP)

# # Introduction
# 
# ## Context
# 
# Exchange rates are a common indicator for both travelers to investors. While these exchange quotes can be readily obtained anywhere, it is much easier for people to gain key insights through simple chart visualisation. 
# 
# ## Objective
# 
# Create a creative Chart with Matplotlib to show the historical change of the major exchange rates.
# 
# ## Importing required packages

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ## Data Preparation
# I will use the exchanges rates EUR, GBP, CHF, CAD, CNY and JPY. You can try this code with every currency pair you want, it's just an example.

# In[ ]:


#Import Data
df = pd.read_csv('../input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv')

#replace NaN Values with previous value
df = df.replace('ND', np.nan)
df = df.fillna(method ='pad')

#select columns with the major exchange rates
df = df[['Time Serie', 'UNITED KINGDOM - UNITED KINGDOM POUND/US$','EURO AREA - EURO/US$' ,'SWITZERLAND - FRANC/US$', 'CANADA - CANADIAN DOLLAR/US$', 'CHINA - YUAN/US$', 'JAPAN - YEN/US$']]


#change data types
df['Time Serie'] = pd.to_datetime(df['Time Serie'], format = '%Y-%m-%d' )
for i in [i for i in list(range(len(df.columns))) if i not in [0]]:
    df.iloc[:,i] = pd.to_numeric(df.iloc[:,i])

#Exponential Smoothing of the time series for a better look
for column in df.drop('Time Serie', axis=1):
    df[column] = df.loc[:,column].ewm(span=30,adjust=False).mean()

df


# ## Generate a Grid with Linecharts
# I was inspirated by: [python-graph-gallery.com](https://python-graph-gallery.com/125-small-multiples-for-line-chart/) 

# In[ ]:


# Initialize the figure
plt.figure(figsize=(19.2,10.8), facecolor = (0.900, 0.900, 0.900))
plt.style.use('seaborn-darkgrid')
 
# create a color palette
palette = plt.get_cmap('Set1')
 
# multiple line plot
num=0
for column in df.drop('Time Serie', axis=1):
    num+=1
 
    # Find the right spot on the plot
    plt.subplot(2,3, num)
 
    # plot every groups, but discreet
    for v in df.drop('Time Serie', axis=1):
        plt.plot(df['Time Serie'], df[v], marker='', color='grey', linewidth=0.6, alpha=0.3)
 
    # Plot the lineplot
    plt.plot(df['Time Serie'], df[column], marker='', color=palette(num), linewidth=2.4, alpha=0.9, label=column)
 
    # Set the Limits
    if df[column].max() < 2:
        y = 2
    else: 
        y = df[column].max() + (df[column].max() / 10)
    plt.ylim(0 ,y)
    
    #plt.margins(0.001,0.1)

    # Not ticks everywhere
    if num in range(4) :
        plt.tick_params(labelbottom=False)
    if num not in [1,4] :
        plt.tick_params(labelleft= False)
    if num in range(4,7):
        plt.tick_params(labelleft= True)
    
    # Add title
    plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )

# general title
plt.suptitle("Major Exchange Rates", fontsize=50, fontweight=0, color='black', weight = 'bold', y=0.98)


# ## Animated Linechart
# I also created an animated Plot with matplotlib's FuncAnimation ([Documentation](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.animation.FuncAnimation.html)).
# You can check it out on YouTube or see it below. 
# 

# In[ ]:


from IPython.display import HTML
HTML('<iframe width="640" height="360" src="https://www.youtube.com/embed/-6aZEfbflMY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


# I hope you get some inspiration for your next Data Visualisation and an good insight in the change of the historical change of the major currencies.
# 
# On [Twitter](https://twitter.com/chartbearx), [Reddit](https://www.reddit.com/user/chartbear) and [YouTube](https://www.youtube.com/channel/UCTlKjrqvuAKAd4jTphXaUfg?sub_confirmation=1) i share data visualisations with a wide range of topics to show you intresting data in small bites.
# 
# Have a nice Day and keep visualize.
