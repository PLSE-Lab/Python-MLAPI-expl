#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


PGA = pd.read_csv('../input/pga-tour-20102018-data/PGA_Data_Historical.csv')


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Determine the data for Rickie Fowler
PGA_RICKIE = PGA[PGA['Player Name']=='Rickie Fowler']
PGA_RICKIE = PGA_RICKIE.drop(['Player Name'], axis =1)


# In[ ]:


#Determine how many TOP 10 with WIN for Rickie since 2010
PGA_RICKIE_TOP10_Win = PGA_RICKIE[PGA_RICKIE['Variable'] == 'Top 10 Finishes - (1ST)']
PGA_RICKIE_TOP10_Win


# In[ ]:


#Determine how many TOP 10 for Rickie since 2010
PGA_RICKIE_TOP10_VAR = PGA_RICKIE[PGA_RICKIE['Variable'] == 'Top 10 Finishes - (EVENTS)']
PGA_RICKIE_TOP10_VAR
#Pretty impressive !?!


# In[ ]:


#Represent these results through a plot
PGA_RICKIE_TOP10_VAR['Season']=PGA_RICKIE_TOP10_VAR['Season'].astype(float)
PGA_RICKIE_TOP10_VAR['Value']=PGA_RICKIE_TOP10_VAR['Value'].astype(float)
plt.plot(PGA_RICKIE_TOP10_VAR['Season'], PGA_RICKIE_TOP10_VAR['Value'], color='red',marker ='o' ,label='Rickie')
plt.title('TOP 10 for Rickie')
plt.xlabel('Season')
plt.ylabel('Number of Top10')
plt.ylim(top=30)
plt.ylim(bottom =0)
#Represent a high level constance !!


# In[ ]:


#What about the driving...
PGA_RICKIE_driving = PGA_RICKIE[PGA_RICKIE['Variable']=='Driving Distance - (AVG.)']
PGA_RICKIE_driving
#Constant over the years


# In[ ]:


#Represent results with a plot
PGA_RICKIE_driving['Season']=PGA_RICKIE_driving['Season'].astype(float)
PGA_RICKIE_driving['Value']=PGA_RICKIE_driving['Value'].astype(float)
plt.plot(PGA_RICKIE_driving['Season'], PGA_RICKIE_driving['Value'], color='red',marker ='o' ,label='Rickie')
plt.title('Driving Distance')
plt.xlabel('Season')
plt.ylabel('Distance in Yds')
plt.ylim(top=310)
plt.ylim(bottom =250)


# In[ ]:


#Driving wich is concentrate around the 290 yds
sns.distplot(PGA_RICKIE_driving['Value'])


# In[ ]:


#And his driving is in a good trend over the seasons
sns.jointplot(x='Season', y='Value', data=PGA_RICKIE_driving, kind= 'reg', ratio=5, color='green', space=0.2,height =10)


# In[ ]:


#What about the driving accuracy
PGA_RICKIE_driving_acc = PGA_RICKIE[PGA_RICKIE['Variable']=='Hit Fairway Percentage - (%)']
PGA_RICKIE_driving_acc


# In[ ]:


#Results with plotting
PGA_RICKIE_driving_acc['Season']=PGA_RICKIE_driving_acc['Season'].astype(float)
PGA_RICKIE_driving_acc['Value']=PGA_RICKIE_driving_acc['Value'].astype(float)
sns.jointplot(x='Season', y='Value', data=PGA_RICKIE_driving_acc, kind= 'reg', ratio=5, color='green', space=0.2,height =10)
#Once again : improvment over the seasons...


# In[ ]:


#What about the Green in Reg
PGA_RICKIE_GreenReg = PGA_RICKIE[PGA_RICKIE['Variable']=='Greens in Regulation Percentage - (%)']
PGA_RICKIE_GreenReg


# #Rickie has everyhing to win more often on the PGA Tour. Stats/figures are talking by themself. So, Go Rickie !!
