#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#We have to add file before starting analyse
data=pd.read_csv("../input/athlete_events.csv")
data1=pd.read_csv("../input/noc_regions.csv")


# In[ ]:


#These methods show us some kind of statistical values and info of data
data.info()  
data.describe() 


# In[ ]:


#Correlation between datas
data.corr()


# In[ ]:


#Correlation map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(),annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#"annot=True" shows us numbers in cells
#"linewidths" shows us line widths between cells
#"fmt" shows that numbers of digit
plt.show()


# In[ ]:


data.head(10) #list of top10


# In[ ]:


data.tail(10) #list of last10


# In[ ]:


data.columns #columns of data


# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
plt.figure(1)
data.Height.plot(kind = 'line', color = 'g',label = 'Height',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
plt.legend()
plt.xlabel("Id")
plt.ylabel("Height")
plt.show()

#The other way to show line plot
plt.figure(2)
plt.plot(data.ID,data.Weight,color="red",label="Weight",alpha=0.6,linestyle="-")
plt.legend()
plt.grid(True,alpha = 0.7)
plt.xlabel("Id")
plt.ylabel("Weight")
plt.show()


# In[ ]:


plt.scatter(data.Weight,data.Height,color="blue")
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()


# In[ ]:


data.Age.plot(kind = 'hist',bins = 30,figsize = (12,12))
plt.show()


# In[ ]:


#plt.hist(data.Age,bins = 30,figsize = (12,12))
#plt.show()


# In[ ]:


data.Age.plot(kind = 'hist',bins = 30,figsize = (12,12))
plt.clf() #removing plot


# In[ ]:


#dictionary has keys and values

dic={"HUN":"Budapest","DEN":"Copenhagen","SLO":"Bratislava"}
print(dic)
print(dic.keys())
print(dic.values())


# In[ ]:


dic["HUN"]="Szeged" #changing value
print(dic)
dic["NOR"]="Oslo" #adding value
print(dic)
del dic["HUN"]  #removing
print(dic)
print("DEN" in dic)  #checking value
dic.clear()          #remove all entries in dict
print(dic)


# In[ ]:


del dic #destroy the dictionary
print(dic) 


# In[ ]:


series = data['Age']        # data[''] = series
print(type(series))    #shows type
data_frame = data[['Age']]  # data[['']] = data frame
print(type(data_frame))


# In[ ]:


#Comparison operators
print(5<3)
print(7>6)
print(4!=5)
print(6==6)


# In[ ]:


a=data["Age"]>70
data[a]

#There are 101 athletes older than 70 years old


# In[ ]:


data[a]["Event"].unique()
#it shows us which sports available for people who older than 70 in Olympic games.


# In[ ]:


b=(data["Age"]>70) & (data["Event"]=="Art Competitions Mixed Sculpturing") 
#filtering
data[b] 


# In[ ]:


#This is the another way to show same as previous one
data[np.logical_and(data["Age"]>70, data["Event"]=="Art Competitions Mixed Sculpturing")]        


# In[ ]:


#10 to 0 countdown 
j=10
while j!=0:
    print(j)
    j=j-1
print(j,"Happy New Year")


# In[ ]:



d = [1,2,3,4,5]
for each in d:
    print(each)
print("--")

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(d):
    print(index," : ",value)
print('--')

# For dictionaries
dic={"HUN":"Budapest","DEN":"Copenhagen","SLO":"Bratislava"}
for key,value in dic.items():
    print(key," : ",value)
print('--')

# For pandas we can achieve index and value
for index,value in data[['Age']][0:8].iterrows():
    print(index," : ",value)


# In[ ]:




