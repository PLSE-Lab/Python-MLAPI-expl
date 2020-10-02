#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import packages 
import pandas as pd
from io import StringIO
import matplotlib as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import datetime
import seaborn as sns


# **Data per day**
# 
# The function mean_per_hour() takes the files as input argument and groups the data per day. 

# In[ ]:


#file paths
flow_sch = "../input/flow_schwartau.csv"
temperature_sch = "../input/temperature_schwartau.csv"
weight_sch = "../input/weight_schwartau.csv"
humidity_sch = "../input/humidity_schwartau.csv"

#group data per day: sum of flow (=netflow) and mean temperature, weight and humidity
def mean_per_hour(ifile):
    name = str(ifile)
    ifile = pd.read_csv(ifile, sep=',', decimal=".")
    ifile['timestamp'] = pd.to_datetime(ifile['timestamp'])
    ifile.sort_values(by="timestamp")  #sort values by date
    ifile.set_index('timestamp', inplace=True) #date as index
    if "flow" in name: 
        ifile = ifile.groupby(pd.Grouper(freq='D')).sum() #group by day and sum
        ifile.ffill() #if NaN fill with value of previous day
    else: 
        ifile = ifile.groupby(pd.Grouper(freq='D')).mean() #group by day and calculates mean
        ifile.ffill() #if NaN fill with value of previous day
    return ifile 

#calling mean_per_hour function on the input files
netflow_sch = mean_per_hour(flow_sch)
temperature_sch = mean_per_hour(temperature_sch)
weight_sch = mean_per_hour(weight_sch)
humidity_sch = mean_per_hour(humidity_sch)


# **Netflow**
# 
# The netflow is positive during the summer months. In the other seasons, the inflow in seems to approach the outflow, therefore the netflow is around 0. 

# In[ ]:


ax = netflow_sch[netflow_sch < 0].plot(title="Flow per hour", color="green")
netflow_sch[netflow_sch > 0].plot(ax=ax, color="orange")
ax.legend(["inflow", "outflow"])

#mark input=output line 
ax.axhline(y=0, color='blue', linestyle='--', linewidth=0.5)


# **Humidity**
# 
# Humidity between the 50% and 60% is optimal for breeding. From the graph it can be noted, that the conditions have been optimal around the summer of 2019. 

# In[ ]:


#seperate low and optimal data
low_humid = humidity_sch[humidity_sch < 50]
optimal_humid = humidity_sch[(humidity_sch > 50) & (humidity_sch < 60)]

#plot all, low and optimal data
ax1=humidity_sch.plot(color="red", alpha = 0.4)
low_humid.plot(ax=ax1, color="yellow")
optimal_humid.plot(ax=ax1, color="green")
ax1.legend(["high","low","optimal"])

#mark optimal boarder in graph
ax1.axhline(y=50, color='blue', linestyle='--', linewidth=0.5)
ax1.axhline(y=60, color='blue', linestyle='--', linewidth=0.5)


# **Temperature**
# 
# Optimal temperature for breeding is between the 30-35 degrees. From the graph it can be noted, that the conditions have been optimal around the summer of 2017, 2018 and 2019. 

# In[ ]:


#seperate optimal data
optimal_temp = temperature_sch[(temperature_sch > 30) & (temperature_sch < 35)]

#plot all and optimal data
ax2=temperature_sch.plot(color="red", alpha = 0.4)
optimal_temp.plot(ax=ax2, color="green")
ax2.legend(["low","optimal"])

#mark optimal boarder in graph
ax2.axhline(y=30, color='blue', linestyle='--', linewidth=0.5)
ax2.axhline(y=35, color='blue', linestyle='--', linewidth=0.5)


# **Weight** 
# 
# In the period of May and June 2019, the weight measurements seem to be off as they approach 0. Overall the weight has seemed to decrease a bit. The weight at the end of summer in 2017 was higher than 2018. This might point at a smaller honey reserve, which could lead to increased death in winter. 
# 

# In[ ]:


#seperate above mean
above_mean = weight_sch[weight_sch > weight_sch["weight"].mean()]

#plot all and above mean data
ax3 = weight_sch.plot(color="red", alpha = 0.4)
above_mean.plot(ax=ax3, color="green")
ax3.legend(["below mean weight","above mean"])

#mark mean
ax3.axhline(y=weight_sch["weight"].mean(), color='blue', linestyle='--', linewidth=0.5)

