#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#import necessary packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
df = pd.read_csv('../input/food_prices_watch.csv') #reading the file
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df.info()


# In[ ]:


timeline = ('Jun-16','Jul-16', 'Aug-16', 'Sep-16','Oct-16','Nov-16','Dec-16','Jan-17','Feb-17','Mar-17','Apr-17','May-17','Jun-17') #this is a list of the labels that will appear in the chart


# In[ ]:





# In[ ]:


#for tomato
tomato = df[df.ItemLabels == 'Tomato'] #getting the data for tomato
tomato = tomato.iloc[:,6:19] #selecting data for June 2016-June 2017
tomato = tomato.values.T.tolist()
len(tomato) == len(timeline) #to check if the two lists have the same length
tomato


# In[ ]:


#for Medium Grain rice
rice_medium = df[df.ItemLabels == 'Rice Medium Grained']
rice_medium = rice_medium.iloc[:,6:19] #selecting data for June 2016-June 2017
rice_medium = rice_medium.values.T.tolist() #converting the data frame into a list
len(rice_medium) = len(timeline)
len(rice_medium) == len(timeline) #to check if the two lists have the same length


# In[ ]:


#for Medium Grain rice
rice_medium = df[df.ItemLabels == 'Rice Medium Grained']
rice_medium = rice_medium.iloc[:,6:19] #selecting data for June 2016-June 2017
rice_medium = rice_medium.values.T.tolist() #converting the data frame into a list
len(rice_medium) == len(timeline) #to check if the two lists have the same length


# In[ ]:


#for white loose garri
white_garri = df[df.ItemLabels == 'Gari white,sold loose']
white_garri = white_garri.iloc[:,6:19] #selecting data for June 2016-June 2017
white_garri = white_garri.values.T.tolist() #converting the data frame into a list
len(white_garri) == len(timeline)


# In[ ]:


#for yam tuber
yam = df[df.ItemLabels == 'Yam tuber']
yam = yam.iloc[:,6:19] #selecting data for June 2016-June 2017
yam = yam.values.T.tolist() #converting the data frame into a list
len(yam) == len(timeline)


# In[ ]:



#plotting the data
x_axis = np.arange(len(timeline))
y1 = tomato
y2 = rice_medium
y3 = white_garri
y4 = yam


# In[ ]:


#garri is flakes made from processed cassava and comes in two varities-Yellow and White


# In[ ]:


plt.xticks(x_axis, timeline, rotation='vertical')
plt.xlabel('Timeline (June 2016-June 2017)')
plt.ylabel('Price per KG)
plt.ylabel('Price per KG')
plt.ylim(150, 450)
plt.title('Changes in price of selected food items')
plt.plot(x_axis,y1, '-r', label='Tomato')
plt.plot(x_axis,y2, '-y', label='Medium Grained Rice')
plt.plot(x_axis,y3, '-b', label='White Garri')
plt.plot(x_axis,y4, '-g', label='Yam Tuber')
plt.legend(loc='upper left')


# In[ ]:





# In[ ]:


plt.xticks(x_axis, timeline, rotation='vertical')
plt.xlabel('Timeline (June 2016-June 2017)')
plt.ylabel('Price per KG')
plt.ylim(150, 450)
plt.title('Changes in price of selected food items')
plt.plot(x_axis,y1, '-r', label='Tomato')
plt.plot(x_axis,y2, '-y', label='Medium Grained Rice')
plt.plot(x_axis,y3, '-b', label='White Garri')
plt.plot(x_axis,y4, '-g', label='Yam Tuber')
plt.legend(loc='upper left')


# In[ ]:




