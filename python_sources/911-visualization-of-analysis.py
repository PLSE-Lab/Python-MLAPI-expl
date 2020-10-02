#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[ ]:


calls = pd.read_csv('../input/911.csv')
calls


# # Question 1 : What are the columns containing the null Values?

# In[ ]:


calls.isnull().sum()


# In[ ]:


# Our data has some missing values that may affect our analysis.


# # Question 2:
# # There are 3 reasons listed for 911 calls in the column 'Title' :
#  # 1.) EMS
#  # 2.) Traffic
#  # 3.) Fire
# # Lets make another column named 'Reason' with this info and see which  is the most common reason for 911 call. 

# In[ ]:


# adding column 'Reason'
calls['Reason'] = calls['title'].apply(lambda x: x.split(':')[0])
calls.head()


# In[ ]:


#reason counts and its plots
df=calls['Reason'].value_counts()
print(df)
sns.countplot(x='Reason',data=calls,palette='viridis')


# # Question 3: What are the top 10 zipcodes with most number of 911 calls?

# In[ ]:


calls['zip'].value_counts().head(10).plot.bar(color = 'black')
plt.xlabel('Zip Codes',labelpad = 22)
plt.ylabel('Number of Calls')
plt.title('Zip Codes with Most 911 Calls')


# In[ ]:


# zipcode 19401 has most number of complaints!!


# # Question 4: What are the top 10 townships with most number of calls?

# In[ ]:


#Visualization of number of calls relative to township

calls['twp'].value_counts().head(10).plot.bar(color = 'orange')
plt.xlabel('Townships', labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Townships with Most 911 Calls')


# # Question 5: What is the main emergency associated with the call?

# In[ ]:


# Let's split the column 'title', make a column 'emergency' and see how many calls were there for what emergency.


# In[ ]:


calls['Emergency'] = calls['title'].apply(lambda x: x.split(':')[1])
calls['Emergency'].value_counts().head(30)


# In[ ]:


#Hence most 911 calls were for 'Vehicle Accident'.


# In[ ]:


#Visualization of top 10 911 Calls
calls['Emergency'].value_counts().head(10).plot.bar(color = 'red')
plt.xlabel('Emergency',labelpad = 22)
plt.ylabel('Number of 911 Calls')
plt.title('Top 10 Emergency Description Calls')


# # Question 6: What are the count of calls per month  for different reasons? 

# In[ ]:


calls['timeStamp'] = pd.to_datetime(calls['timeStamp'])               # coverting from strings to datetime object
calls['Month'] = calls['timeStamp'].apply(lambda time: time.month)    # creating column"Month'
sns.countplot(x='Month',data=calls,hue='Reason',palette='nipy_spectral')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
byMonth = calls.groupby('Month').count()
byMonth.head(12)


# # Question 7:  Visualising linear relationship for calls per month
# 

# In[ ]:


byMonth['twp'].plot()


# In[ ]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# # Time Analysis: 911 calls between time periods.

# In[ ]:


calls['hour'] = calls['timeStamp'].map(lambda x: x.hour)

groupByMonthDay = calls[(calls['hour'] >= 8) & (calls['hour'] <= 18)].groupby('Month',as_index = False).sum()

yy = groupByMonthDay['e'].values
labels  = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov']
xx = groupByMonthDay['Month'].values
width = 1/1.5
plt.bar(xx, yy, width, color="black",align='center')
plt.title('911 Calls each month 8 am to 6 pm')
plt.xticks(xx, labels)
plt.show()


# In[ ]:


groupByMonthNight = calls[(calls['hour'] > 18) | (calls['hour'] < 8)].groupby('Month',as_index = False).sum()

groupByMonthNight.head()

y = groupByMonthNight['e'].values
labels  = ['Jan','Feb','Mar','Apr','May','June','July','Aug','Sep','Oct','Nov']
x = groupByMonthNight['Month'].values
width = 1/1.5
plt.bar(x, y, width, color="blACK",align='center')
plt.title('911 Calls each month 6 pm to 8 am')
plt.xticks(x, labels)
plt.show()

