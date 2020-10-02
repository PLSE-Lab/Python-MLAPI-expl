#!/usr/bin/env python
# coding: utf-8

# fill fill fill

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import GeocoderDotUS
from mpl_toolkits.basemap import Basemap

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


Nashville = pd.read_csv(os.path.join('../input', 'Nashville_housing_data_2013_2016.csv'))
Nashville.info()


# In[ ]:


# split the data based on year of sale. 
Nashville['Sale Year'] = Nashville['Sale Date'].str[:4]
Nashville['Month'] = Nashville['Sale Date'].str[5:7]
Nashville['Day'] = Nashville['Sale Date'].str[8:10]
Nashville['Sale Month'] = Nashville.Month.map({'01': 'Jan', '02': 'Feb', '03': 'Mar', '04':'Apr','05': 'May',
                                          '06': 'Jun','07': 'Jul','08': 'Aug','09': 'Sep','10': 'Oct',
                                          '11': 'Nov','12': 'Dec'})
Nashville['Sale Day'] = Nashville.Day.astype(float)
Nashville.drop('Month', axis=1)
Nashville.drop('Day', axis=1)

fig, (axis1, axis2, axis3) = plt.subplots(1,3, figsize=(12,5))
sns.countplot(x='Sale Year', data=Nashville, ax = axis1)
sns.countplot(x='Sale Month', data=Nashville, ax = axis2)
Nashville['Sale Day'].hist(bins = 15, ax= axis3)


# At first glance, a couple things pop out: sales tend to be concentrated in the middle of the year and at the end of the month. 

# In[ ]:


#Nashville['Property Address']
#TODO: get geopy to work in kernel

#geolocator = GeocoderDotUS(format_string="%s, Nashville, TN")
#address, (latitude, longitude) = geolocator.geocode("1707 West End Ave")
#print(address, latitude, longitude)


# In[ ]:


Nashville['Property City'].value_counts()
#sns.countplot(x='Property City', data=Nashville)


# In[ ]:


plt.hist(Nashville['Sale Price'],20)


# In[ ]:


Nashville['log_sale_price'] = np.log(Nashville['Sale Price'])
plt.hist(Nashville['log_sale_price'],8)
#Nashville['Sale Price'].describe()


# A sale for $54 million?

# In[ ]:


Nashville.loc[Nashville['Sale Price'] == 54278060]


# Note to self:
# group apartment bldgs?

# In[ ]:


Nashville['Land Use'].value_counts()


# In[ ]:


duplicate_address = Nashville.loc[(Nashville['Property Address'].duplicated())]
duplicate_address

