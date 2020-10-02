#!/usr/bin/env python
# coding: utf-8

# # Quick analysis of COVID-19 Data
# The data was pulled from the <a href='https://coronavirus.jhu.edu/map.html'>Johns Hopkins Center for Systems Science and Engineering (CSSE)</a>
# 
# Data was captured from their github page (https://github.com/CSSEGISandData/COVID-19) on 4/7. You can run this Jupyter Notebook to pull the latest data and plot it.

# First, import required python libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# Now pull the US data. The US data is broken down by state and county.
# After download of the csv file into a pandas dataframe, show the beginning of the file.

# In[ ]:


df_us = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
df_deaths_us = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
df_recovered_us = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')

df_deaths_us.head()


# Then, read the global data, which is broken down by country.

# In[ ]:


df_global = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_deaths_global = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df_recovered_global = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df_global.head()


# In[ ]:


covid_data = {'global_cases':df_global, 'global_deaths':df_deaths_global, 'global_recovered':df_recovered_global,'us_cases':df_us, 'us_deaths':df_deaths_us, 'us_recovered':df_recovered_us }


# Define a function that takes a dataframe and matches a value for the column name to a country/state name to extract the number of confirmed cases and creates a few plots of the time series data.

# In[ ]:


def plot_data(data,column_name,column_val):
    if column_name=='Country/Region':
        dataframe = data['global_cases']
        deaths = data['global_deaths']
    elif column_name=='Province_State':
        dataframe = data['us_cases']
        deaths = data['us_deaths']
        
    cc = dataframe[dataframe[column_name]==column_val].iloc[:,11:].T.sum(axis = 1)
    cc = pd.DataFrame(cc)
    cc.columns = ['Cases']
    cc = cc.loc[cc['Cases']>0]

    y = np.array(cc['Cases'])
    x = np.arange(cc.size)
    dy = np.append([0],np.diff(y))
    
    cd = deaths[deaths[column_name]==column_val].iloc[:,12:].T.sum(axis = 1)
    cd = pd.DataFrame(cd)
    cd.columns = ['Deaths']
    cd = cd.loc[cd['Deaths']>0]

    yd = np.array(cd['Deaths'])
    xd = np.arange(cd.size)
    dd = np.append([0],np.diff(yd))
    
    plt.figure(1,figsize=(15,15))
    ax1=plt.subplot(221)
    ax1.semilogy(x,y)
    plt.title(column_val+' Number of cases (semilog)')
    plt.xlabel('days')
    plt.ylabel('Cumulative Cases (log)')
    ax1.minorticks_on()
    ax1.grid(which='minor',linestyle=':')
    ax1.grid(which='major',linestyle='-')
       
    ax2=plt.subplot(222)
    ax2.bar(x,dy)
    plt.title(column_val+' Daily increase in cases (linear)')
    plt.xlabel('days')
    plt.ylabel('Change in cases')
    ax2.grid()
    
    try:
        ax3=plt.subplot(223)
        ax3.loglog(y,dy,'b.')
        plt.title(column_val+' Cumulative cases vs Change in cases (loglog)')
        plt.xlabel('Cululative cases (log)')
        plt.ylabel('Change in cases (log)')
        ax3.minorticks_on()
        ax3.grid(which='minor',linestyle=':')
        ax3.grid(which='major',linestyle='-')
    except:
        print(y.size,dy.size)
        
    ax4=plt.subplot(224)
    ax4.bar(xd,dd)
    plt.title(column_val+' Daily increase in deaths (linear)')
    plt.xlabel('days')
    plt.ylabel('Change in deaths')
    ax2.grid()
        


# Let's first look at the South Korea total numbers.
# 
# The first plot is total cases versus time on a semilog plot. If this curve is linear, then the cases are growing exponentially. This exponential growth can't be sustained indefinitely. Eventually the whole population will be infected, if unchecked and the number of cases will stop rising. Measures like social distancing and quarantine with or without contact tracing will also stop the growth. The curve will actually be a logistic curve in the end.
# 
# The slope of the semilog plot represents the exponent $n$ of the exponential model:
# 
# $c = c_0e^{nt}$
# 
# since, taking the log, we have a linear plot in time with the slope as $n$:
# 
# $ln(c) = nt+ln(c_0)$
# 
# To find the time to double, we use $2c_0 = c_0e^{nt}$ and solve for t:
# 
# $t_d = ln(2)/n$
# 
# The second plot below shows daily increase in cases.
# 
# A useful plot to determine if you are starting to make a difference in slowing or stopping the spread is to plot a loglog plot with total cases on the x axis and new cases on the y axis. The time aspect on this plot is inferred, since it is just a scatter plot of these data points over time. 
# 
# On the plot of cases for South Korea, you can see that they were able to slow growth by noting the 'knee' in the curve when new cases started to slow down.

# In[ ]:


plot_data(covid_data,'Country/Region','Korea, South')


# Now, looking at Italy, we can see that they have started to make a difference, but new cases has not gone to zero yet.

# In[ ]:


plot_data(covid_data,'Country/Region','Italy')


# China's data is suspect, but here are the official numbers. The loglog plot looks very different than other countries'. This is a tip-off that something is funny with the data.

# In[ ]:


plot_data(covid_data,'Country/Region','China')


# In the US, there is still a long way to go as of 4/7, but the slope has started to decrease.
# 
# By the way, the number of days for cases to double is inversely proportional to the slope of the semilog curve.

# In[ ]:


plot_data(covid_data,'Country/Region','US')


# Now, let's look at some individual states, starting with the state with the most cases, NY. It looks like the curve is bending now.

# In[ ]:


plot_data(covid_data,'Province_State','New York')


# How about NJ? It's bending, too.

# In[ ]:


plot_data(covid_data,'Province_State','New Jersey')


# Washington's data is kind of noisy, since there are fewer data points, but growth has definitely been slower than NY/NJ.

# In[ ]:


plot_data(covid_data,'Province_State','Washington')


# Let's check out California. It looks like they have started bending slightly, but the data is a bit jumpy.

# In[ ]:


plot_data(covid_data,'Province_State','California')


# Florida looks like it is on the decline despite their delay in instituting stay-at-home policies. Maybe that's a sign that the disease is declining everywhere and not just in places that have strict lock downs. That would be a good thing...

# In[ ]:


plot_data(covid_data,'Province_State','Florida')


# In[ ]:


plot_data(covid_data,'Country/Region','Sweden')


# In[ ]:


plot_data(covid_data,'Country/Region','Japan')


# In[ ]:


plot_data(covid_data,'Country/Region','Russia')


# In[ ]:


plot_data(covid_data,'Country/Region','Turkey')


# In[ ]:




