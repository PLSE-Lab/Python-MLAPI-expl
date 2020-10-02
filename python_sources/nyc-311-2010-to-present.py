#!/usr/bin/env python
# coding: utf-8

# **1.  Import a 311 NYC service request**

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


#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Import a 311 NYC service request
df = pd.read_csv('../input/311_Service_Requests_from_2010_to_Present.csv')


# **2.  Basic data exploratory analysis **

# In[ ]:


#Shape dataset before dropping nan values
df.shape


# In[ ]:


#Display column names
df.columns


# In[ ]:


#Status of tickets
df['Status'].value_counts().plot(kind='bar',alpha=0.6,figsize=(5,5))
plt.show()


# In[ ]:


#To get nan values in the entire dataset
df.isnull().sum()


# In[ ]:


#fix blank values in City column
df['City'].dropna(inplace=True)


# In[ ]:


#fix blank values in Complaint Type column
df['Complaint Type'].dropna(inplace=True)


# In[ ]:


#To get nan values in the entire dataset
df.isnull().sum()


# In[ ]:


#Group dataset by complaint type to display
groupedby_complainttype = df.groupby('Complaint Type')


# In[ ]:


#Find the top complaint type 
groupedby_complainttype['Complaint Type'].value_counts().nlargest(1)


# In[ ]:


#Group dataset by major complaint type to display
grp_data = groupedby_complainttype.get_group('Blocked Driveway')
grp_data.shape


# In[ ]:


#count of null values in grouped city column data
grp_data['City'].isnull().sum()


# In[ ]:


#count of null values in grouped Location Type column data
grp_data['Location Type'].isnull().sum()


# In[ ]:


#count of null values in grouped Complaint Type column data
grp_data['Complaint Type'].isnull().sum()


# In[ ]:


#fix those NAN with "unknown city" value instead
grp_data['City'].fillna('Unknown City', inplace =True)


# In[ ]:


#fix those NAN with "Location Type" value instead
grp_data['Location Type'].fillna('Unknown Location', inplace =True)


# In[ ]:


#Scatter plot displaying all the cities that raised complaint of type 'Blocked Driveway'
plt.figure(figsize=(20, 15))
plt.scatter(grp_data['Complaint Type'],grp_data['City'])
plt.title('Plot showing list of cities that raised complaint of type Blocked Driveway')
plt.show()


# In[ ]:


#Display array of Location Types
df['Location Type'].values


# In[ ]:


#Display complaint type and city together
df[['Complaint Type','City']]


# **3.  Find major complaint types**

# In[ ]:


#Find the top 10 complaint types 
groupedby_complainttype['Complaint Type'].value_counts().nlargest(10)


# In[ ]:


#Plot a bar graph of count vs. complaint types
df['Complaint Type'].value_counts().plot(kind='barh',alpha=0.6,figsize=(5,5))
plt.show()


# **4.  Visualize the complaint types**

# In[ ]:


#Display the major complaint types and their count
groupedby_complainttype['Complaint Type'].value_counts().nlargest(50)


# In[ ]:





# In[ ]:




