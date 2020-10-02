#!/usr/bin/env python
# coding: utf-8

# **Automation of Control Charts Using Python**
# 
# Using dummy time-series data, Python will compute standard deviation and control limits. Then the data is graphed using matplotlib. Next steps are to write a function that takes data input for any process and aggregates charts monthly. A function is necessary because there will be different control parameters and nuances (e.g. shifts, trends, etc.) for different processes. I would like to standardize the script as much as possible.
# 
# oneill.evan.r@gmail.com
# 

# In[ ]:


## IMPORT PACKAGES

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


## READ CONTROL CHART DATA

raw = pd.read_csv('../input/test_data.csv')
arr = np.array(raw)


# In[ ]:


## ADD CALCULATED COLUMNS

ratio = np.array((arr[:,2]/arr[:,3])*100).reshape(-1,1)
vals = np.hstack((arr,ratio))


# In[ ]:


## DETERMINE CHART PARAMETERS

center = np.mean(vals[:,-1])
print('Mean is: ' + str(center))
sd = np.std(vals[:,-1])
print('Standard deviation is: ' + str(sd))
lim_upper = center + sd*3
print('Upper control limit is: ' + str(lim_upper))
lim_lower = center - sd*3
print ('Lower control limit is: ' + str(lim_lower))
mnth = vals[:,1]
pct = vals[:,-1]


# In[ ]:


## CREATE PLOT

plt.clf()
plt.figure(2, figsize = [15,10])
plt.grid(which = 'major', axis = 'y')
data = plt.scatter(mnth, pct, marker = 'D', color = 'b')
ucl = plt.axhline(lim_upper, linestyle = '--', color = 'r', linewidth = 2)
ctr = plt.axhline(center, color = 'r', linewidth = 2)
lcl = plt.axhline(lim_lower, linestyle = '--', color = 'r', linewidth = 2)
plt.xticks(rotation = 90)
plt.ylim(0,120)
plt.legend((data, ctr, ucl),('Monthly Percent Compliance','Baseline Averages','Control Limits'), loc = 'center')
plt.show()


# In[ ]:




