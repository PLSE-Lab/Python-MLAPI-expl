#!/usr/bin/env python
# coding: utf-8

# # Come With Me If You Want To Lift
# 
# 
# No one likes going to a crowded gym.  You have to wait to use equipment, the gym is loud, and might smell bad.  Its a good guess that early morning is a good time to go, but if you're like me, you can't get up that early.
# 
# What if we could visualize what times and days were the busiest?  We can.  The data is in seconds after the start of the day, so let's convert that to hours.  Then, we can create a heat map from the data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/data.csv')


# In[ ]:


#analyze time in hours instead of seconds
df['Hour'] = df.timestamp.apply( lambda x: int(np.floor(x/3600))) 

g = df[['Hour','number_people','day_of_week']]

#Group by tme and day
F = g.groupby(['Hour','day_of_week'], as_index = False).number_people.mean().pivot('day_of_week','Hour', 'number_people').fillna(0)


grid_kws = {"height_ratios": (.9, .05), "hspace": .3}

dow= 'Monday Tuesday Wednesday Thursday Friday Saturday Sunday'.split()
dow.reverse()

ax = sns.heatmap(F, cmap='RdBu_r',cbar_kws={"orientation": "horizontal"})
ax.set_yticklabels(dow, rotation = 0)
ax.set_ylabel('')
ax.set_xlabel('Hour')

cbar = ax.collections[0].colorbar
cbar.set_label('Average Number of People')


# Our intuition is pretty good.  Not much activity in the early morning.  
# 
# Other interesting insights:
# 
# -Friday at 5 you can see the 'Friday Night Pump'.  When I was in university, my roommates and I would joke about going to the gym to squeak in a few curls and chest presses.
# 
# -The opposite effect on a Tuesday.  It is eerily vacant at 5pm compared to the rest of the week at the same time.
# 
# -No late night lifters on Friday Saturday.  That is expected of a University Gym

# **BONUS**:  What are the rates of change for each day?  This will show us when the gym is getting busy, and when it is slowing down. 
# 
# We can the gradient to calculate the partial derivative of this scalar field.

# In[ ]:


lwise = np.gradient(F, edge_order = 2)[1]
Fp = pd.DataFrame(lwise, columns=F.columns, index = F.index)


ax = sns.heatmap(Fp, cmap='RdBu_r',cbar_kws={"orientation": "horizontal"})
ax.set_yticklabels(dow, rotation = 0)
ax.set_ylabel('')
ax.set_xlabel('Hour')

cbar = ax.collections[0].colorbar
cbar.set_label('Rate of Change')


# Best times to go to the gym are around 8 am or 10am. Not too many people, and no one is on their way in.  If you go around 1 pm or 2 pm, get in and get out because there are a lot of people on their way.

# In[ ]:




