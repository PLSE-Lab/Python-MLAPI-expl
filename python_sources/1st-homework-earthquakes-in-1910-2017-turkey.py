#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dt = pd.read_csv("../input/earthquake.csv")


# In[ ]:


dt.describe()


# In[ ]:


dt.info() #no longer i dont need to write data.columns 
dt.describe()


# In[ ]:


dt.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(10, 10))
sn.heatmap(dt.corr(), annot=True, linewidths=.9, fmt= '.2f',ax=ax)
plt.show()


# In[ ]:


dt.depth.plot(kind="line",grid=True,label="depth",linestyle=":",color="r")
dt.richter.plot(kind="line",grid=True,label="depth",linestyle="-",color="g")
plt.legend(loc="best")  #this will set the best place for label
plt.title("depth-richter")
plt.show()
#plot of depth's change to richter


# This is depth change for all areas. 

# In[ ]:


eldenizli = dt[dt.area =="eldenizli"]
cavusoglu = dt[dt.area =="cavusoglu"]
ilikaynak = dt[dt.area == "ilikaynak"]

#i selected some areas to examine 


# In[ ]:


plt.plot(eldenizli.richter,eldenizli.depth,label="ELDENIZLI",color="red") 
plt.legend() #this will set the best place for label
ax.set_xlabel("Time")
ax.set_ylabel("Speed")
plt.show()


# Here is Eldenizli's earthquake depths' change to richter
# 
# Now let's compare Eldenizli, Ilikaynak and Cavusoglu's earthquake depths

# In[ ]:


plt.plot(eldenizli.richter,eldenizli.depth,color="yellow",label="eldenizli")
plt.plot(ilikaynak.richter,ilikaynak.depth,color="blue",label="ilikaynak")
plt.plot(cavusoglu.richter,cavusoglu.depth,color="red",label="cavusoglu")
plt.legend()
plt.title("richter-depth comparison")
plt.show()


# In[ ]:


dt.plot(kind='scatter', x='id', y='depth',alpha = 0.5,color = 'red')
plt.title("richter-depth scatter plot")


# This is scatter plot for all areas.
# now i will make it for specific areas

# In[ ]:


plt.scatter(eldenizli.richter,eldenizli.depth,color="yellow",label="eldenizli")
plt.scatter(ilikaynak.richter,ilikaynak.depth,color="blue",label="ilikaynak")
plt.scatter(cavusoglu.richter,cavusoglu.depth,color="red",label="cavusoglu")
plt.legend()
plt.title("scatter plot")
plt.show()  
#It is really similar with making line plot!!


# In[ ]:


##HERE IS A HISTOGRAM OF ELDENIZLI'S EARTHQUAKE DEPTH
plt.hist(eldenizli.depth,bins=50)
plt.xlabel("depth value")
plt.ylabel("frequency")


# I want to find areas which are richter scales are more than average. 

# In[ ]:


avrg = np.mean(dt.richter) #mean
print(avrg)
filter1 = dt.richter > avrg
print(filter1) #if richter scale > mean this code prints True/ else False


# In[ ]:


dt[filter1]  #and this code prints all features if their richter scale > mean

