#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 


import matplotlib.pyplot as plt  # Data visualization
import numpy as np
import pandas as pd # CSV file I/O, e.g. the read_csv function
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


crash_data = pd.read_csv("../input/3-Airplane_Crashes_Since_1908.txt")
# Any results you write to the current directory are saved as output.


def processLocation(df):
   df.Location = df.Location.apply(lambda x: x.split(",")[-1])
   return df


def removeMissindFields(df):
    df = df.dropna(subset=['Location','Date','Operator'])
    return df

def removeFeatures(df):
    df = df.drop(['Route','Time', 'cn/In','Registration','Summary','Ground','Aboard','Flight #','Operator','Type'],axis =1)
    return df
    
crash_data = removeMissindFields(crash_data)
crash_data = removeFeatures(crash_data)
crash_data = processLocation(crash_data)  




# In[ ]:


# count the countries with number of creashes using value_counts() function
crash_count = crash_data['Location'].value_counts()

# change the seried to dataframe for easier plotting of graphs
D1 = crash_count.to_frame().reset_index()
D1.columns = ['Location', 'Ratings']

#change the datatype from object to float for graph plotting 
D1[['Ratings']] = D1[['Ratings']].astype(float)

# save only top 10 values from dataframe
D2 = D1.head(20)  #get the 10 most dangeroud countries to fly

#plot the graph
sns.barplot(D2["Location"], D2["Ratings"], palette="Set3")
plt.xticks(rotation=90)
plt.show()

