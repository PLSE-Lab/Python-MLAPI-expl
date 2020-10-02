#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #basic library for plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#silence warnings
#
import warnings
warnings.filterwarnings('ignore') 

# Any results you write to the current directory are saved as output.






# # Introduction
#    In this notebook I'll be performing exploratory data analysis on our dataset and seeing if there are any patterns associated with the number of violations that occur over time.

# In[ ]:


#Load the data into a data frame
data=pd.read_csv("../input/Red_Light_Camera_Violations.csv")


# In[ ]:


data.info()


# In[ ]:


#Convert VIOLATION DATE to a datetime object for sorting
data["VIOLATION DATE"]=pd.to_datetime(data["VIOLATION DATE"],format='%m/%d/%Y')


# In[ ]:


#Get rid of violations that were documented without a camaera id (<1%)
cameras=data[data['CAMERA ID'].notnull()]
cameras['ID_STRING']=cameras['CAMERA ID'].apply(lambda x: str(x))
cameras.head()


# In[ ]:


#Looking at the violaton counts for each camera
cameras.groupby(["INTERSECTION","CAMERA ID"]).sum()['VIOLATIONS'].describe()


# In[ ]:


#There seems to be two violations at two different intersections by the same camera on two days
indices=cameras.groupby(["CAMERA ID","VIOLATION DATE"]).count()['INTERSECTION'] > 1
temp=cameras.groupby(["CAMERA ID","VIOLATION DATE"]).count()
temp[indices]


# In[ ]:


cameras['Weekday']=cameras['VIOLATION DATE'].apply(lambda x:x.weekday())
cameras['Month']=cameras['VIOLATION DATE'].apply(lambda x:x.month)
cameras['Year']=cameras['VIOLATION DATE'].apply(lambda x:x.year)


# In[ ]:


fig, axes=plt.subplots(nrows=1,ncols=3)
fig.set_figheight(7)
fig.set_figwidth(25)
cameras.groupby('Year').sum()['VIOLATIONS'].plot(ax=axes[0],title="Total violations for each Year")
cameras.groupby('Month').sum()['VIOLATIONS'].plot(ax=axes[1],title="Total violations for each Month")
cameras.groupby('Weekday').sum()['VIOLATIONS'].plot(ax=axes[2],title="Total violations on each day of the week")


# # Remarks
# 
# * 2016 Had the highest amount of violations of any year
# * More of the violations occured during summer months (months 7 and8) than other times of the year 
# * More of the violations occured during weekend days than other days, (Fri-Sun, Days 4-6 respectively)

# In[ ]:


#Plot Of how many violations occured on each day over the past 3 years
vls_per_day=cameras.groupby("VIOLATION DATE", sort=True).sum()['VIOLATIONS']

vls_per_day.plot.line(figsize=(16,3),lw=0.5,title="Total Violations Per Day")


# # Remarks
# 
# * It seems the amount of violations seems to fluctuate quite nicely in a sinusoidal manner
# * The smaller fluctuations may be the result of higher violation counts on the weekend
# * The larger fluctuations may be seasonal, i.e. summer months have more violations than other times of the year
# * We can also see that the plot fluctuates around higher values during the year of 2016

# In[ ]:


#calculate total amount of entries
length=len(vls_per_day)
#get range of dates
dates=cameras.groupby("VIOLATION DATE", sort=False).sum().index

#subtract the max range in days from how many entries we have
length -(dates.max()-dates.min()).days

#Length is 1 which means there are entries for every day (since length is counting which includes the initial starting point date so it shouldn't be 0)


# In[ ]:


fig, Axarray =plt.subplots(ncols=2,nrows=1,figsize=(15,3))
_ =Axarray[0].hist(data['LONGITUDE'][data['LONGITUDE'].notnull()],bins=50)
_ =Axarray[1].hist(data['LATITUDE'][data['LATITUDE'].notnull()],bins=50)


# # Things to explore further
# -- 
# * Perform fourier transforms on vilolations per day to see if there are any more patterns of higher or lower violations
# * plot heat map on chicago area of total violations (or possibly some other metric) 

# In[ ]:





# In[ ]:





# In[ ]:




