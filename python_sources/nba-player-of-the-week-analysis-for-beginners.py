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


# First of all, I chose a dataset from hottest datasets of kaggle which is "NBA player of the week".
# I think this is a very interesting dataset especially for whom likes basketball.
# So lets import the dataset.
# 

# In[ ]:


data = pd.read_csv('../input/NBA_player_of_the_week.csv')


# Secondly, it's good to see what we have in our dataset. So, lets call our first 15 rows from dataset.

# In[ ]:


data.head(15)


# And also we can call dataset information property of python.

# In[ ]:


data.info()


# Yes, it's amazing. Now we can see some information about our data. But I think that's not enough for us.
# I want to see a little bit more information. May be some description of our data set. So we can use description property of python.

# In[ ]:


data.describe()


# By this description, we can see some important calculations about our dataset like min, max values and mean statistics.

# Then, what are columns of our dataset?

# In[ ]:


data.columns


# And, it's really very important to see the degrees of correlation between features of dataset.
# To do this, we can use "data_frame_name.corr( )" property of python and then we can plot it as a heatmap by using matplotlib.

# In[ ]:


data.corr()


# There are some strong positive correlation relations between some of the features in data set.
# For example, we may intuitively say that there is a positive correlation between "Season short" and "Draft Year" by looking at the above table. 
# So, lets see the heatmap of this table. (And to do this, we should import matplotlib and seaborn libraries)

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns  

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# Histograms are good to see frequency of numeric features. 
# So lets see the age frequency of the NBA player of the weeks by using histogram.

# In[ ]:


data.Age.plot(kind = 'hist',bins = 100,figsize = (15,15))
plt.show()


# We can see from the above table that most of the NBA player of the week was 25 years old at that time.  

# We can also use scatter plots for analysing this dataset. 
# We learned some information about correlations between the features of our dataset above.
# So lets see this relation with scatter plots as a visualization example.

# In[ ]:



data.plot(kind='scatter', x='Age', y='Seasons in league',alpha = 0.5,color = 'red', figsize=(12,12))
plt.xlabel('Age')             
plt.ylabel('Seasons in league')
plt.title('Age & Seasons in league Scatter Plot')        
plt.show()


# I think its'not surprising. : ) 
# 
# But this is a good example of strong positive correlation between two features. No outliers, no gaps and no messy dots. 
# 

# And what we could do if we want to create line plot by using this dataset?
# I really wonder your comments?

# Now, I want to filter our dataset by defining some variables.

# In[ ]:


x = data['Age']>35    
data[x]
# We can see only the players older than 35 when he was chosen player of the week by using this filter.


# In[ ]:


y = data['Team']=='Los Angeles Lakers'   
data[y]
# We can see the players from only Utah Jazz Lakers by using this filter.


# In[ ]:


# We can also create new data_frames by using this data.
data_frame_specialities_of_players = data[['Player','Age','Weight','Height']]  
print(type(data_frame_specialities_of_players))
data_frame_specialities_of_players.head()


# Yes, we can do a lot of things by using python.
# But, I think it's enough for our first tutorial.
# 

# In[ ]:





# In[ ]:




