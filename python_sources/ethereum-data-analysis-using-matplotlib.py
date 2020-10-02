#!/usr/bin/env python
# coding: utf-8

# # Import required modules for the analysis and visualizations

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from matplotlib import style
style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read the ethereum data using pandas
# Data is in CSV format,So using read_csv('') function of Pandas to import the data.

# In[ ]:


data = pd.read_csv('../input/all_data.csv')


# In[ ]:


data.info()
print('-'*90)
print(data.head())


# # Total_eth_growth and market-cap-value are columns with same values so we will drop one of them
# 

# In[ ]:


data = data.drop(['total_eth_growth'],axis=1)


# # Convert Unix time stamp to normal time 
# 

# In[ ]:


dateconv = np.vectorize(dt.datetime.fromtimestamp)
data['timestamp'] = dateconv(data['timestamp'])
print(data.head())


# # Plot the each column against date
# 

# In[ ]:


plt.plot(data['timestamp'],data['price_USD'])
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.show()


# We can see the spike in the Ether prices during  June ,2017

# In[ ]:


plt.plot(data['timestamp'],data['blocksize'])
plt.xlabel('Date')
plt.ylabel('blocksize')
plt.show()


# In[ ]:


plt.plot(data['timestamp'],data['hashrate'])
plt.xlabel('Date')
plt.ylabel('hashrate')
plt.show()


# In[ ]:


plt.plot(data['timestamp'],data['total_addresses'])
plt.xlabel('Date')
plt.ylabel('total_addresses')
plt.show()


# In[ ]:


plt.plot(data['timestamp'],data['transactions'])
plt.xlabel('Date')
plt.ylabel('transactions')
plt.show()


# #### **Making subplots**
# I will show you 3 ways of making subplots.
#  1. subplot2grid ,
#  2. add_subplot ,
#  3. subplots ,
# I am creating 3 subplots in a single page with common sharing x-axis using 3 methods mentioned above. 

# 1 -- ***subplot2grid***
# 
# 
# ----------
# 
#  - first parameter (No. of rows,No.of columns) or (No. of divisions on y-axis,No of divisions on x-axis) i.e (3,1).
#  - second parameter (from where the graph should begin) i.e (0,0).
#  - third parameter no.of rows the graph should span.
#  - fourth parameter no. of columns the graph should span.

# In[ ]:


ax1 = plt.subplot2grid((3,1),(0,0),rowspan=1,colspan=1)
ax2 = plt.subplot2grid((3,1),(1,0),rowspan=1,colspan=1)
ax3 = plt.subplot2grid((3,1),(2,0),rowspan=1,colspan=1)


# 2 -- ***add_subplots***
# 
# 
# ----------
# 
# 
#     Here we have to mention size the size and position of each subplot
#     In ax1 - 311 means, figure is divided into 3 rows and 1 column  and 1 is to say that ax1 is first one.
#     In ax2 - 312 means, figure is divided into 3 rows and 1 column  and 1 is to say that ax2 is second one.
#     In ax3 - 313 means, figure is divided into 3 rows and 1 column  and 3 is to say that ax3 is second one

# In[ ]:


fig= plt.figure()
ax1= fig.add_subplot(311)
ax2= fig.add_subplot(312)
ax3= fig.add_subplot(313)


# 3 -- ***subplots***
# 
# 
# ----------
# 
#  - No. of rows in subplot
#  - No. of cols in subplot
#  - whether x axis is shared among them or not
#     

# In[ ]:


fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex= True) 


# In[ ]:


ax1.plot(data['timestamp'],data['price_USD'])
ax2.plot(data['timestamp'],data['total_addresses'])
ax3.plot(data['timestamp'],data['blocksize'])
plt.show()


# In[ ]:


fig2,(ax1,ax2,ax3) = plt.subplots(3,1,sharex= True) 
ax1.plot(data['timestamp'],data['hashrate'])
ax2.plot(data['timestamp'],data['transactions'])
ax3.plot(data['timestamp'],data['market-cap-value'])
plt.show()

