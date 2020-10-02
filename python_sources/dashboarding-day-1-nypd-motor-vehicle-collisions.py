#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


dataset = pd.read_csv('../input/nypd-motor-vehicle-collisions.csv')
dataset.head()


# In[ ]:


dataset.tail()


# In[ ]:


dataset.info()


# This shows all the related data as to how much memory is used, total no. of columns and datatypes for each column, etc.

# * It can be verified through above *head()* and *tail()* data  that the dataset is given in the order of latest to earliest incidents. It can also ve verified from the summary on right.
# * Let's get some statistical insights into the data of people killed and injured like mean.

# In[ ]:


dataset.describe()


# Here it can be seen that the average no. people killed in each case and overall too is very low than that of injured people. Hence, it is quite relieving.
# 
# For instance take the pedestrians:
# > the injured people average is in terms of e-02 while 
# > killed people average is in terms of e-04. Hence, when we would sum up this data, the overall no. of people killed will also be very less than the average no. of people injured.
# 
# Let's try to represent it graphically as no. of people killed and no. of people injured per **BOROUGH**

# In[ ]:


# Filtering the dataframe per BOROUGH
injured_people = dataset[dataset['NUMBER OF PERSONS INJURED'] > 0]
killed_people = dataset[dataset['NUMBER OF PERSONS KILLED'] > 0]

# Plotting the dataframe
fig = plt.figure(figsize=(15, 10))

ax = fig.add_subplot(111)
ax2 = ax.twinx()
injured_people.BOROUGH.value_counts().plot(kind='bar', width = 0.4, color='blue', position=0, ax=ax)
killed_people.BOROUGH.value_counts().plot(kind='bar', width = 0.4, color='red', position=1, ax=ax2)
ax.set_title('Number of persons injured and killed per borough', fontsize=20, fontweight='bold')
ax.set_ylabel('Number of persons injured')
ax2.set_ylabel('Number of persons killed')
plt.show()


# **Red Bar:  No. of People Killed**
# 
# **Blue Bar: No. of People Injured**
