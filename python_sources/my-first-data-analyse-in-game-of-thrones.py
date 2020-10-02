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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Importing CSV file
# 
# 

# In[ ]:


GoTData = pd.read_csv("../input/character-deaths.csv")


# In[ ]:


GoTData.info()


# In[ ]:


GoTData.describe() #Mathematical statistics about the data.


# In[ ]:


GoTData.corr() # We can see in this code's output that Book of Death and Death Year are directly proportional.


# Visualization of correlation

# In[ ]:


f,ax = plt.subplots(figsize=(17,17))
sns.heatmap(GoTData.corr(),annot=True,linewidths=.5,fmt = '.1f',ax=ax)
plt.show()


# First 6 values of the dataset
# 

# In[ ]:


GoTData.head(6)


# Last 6 values of the dataset

# In[ ]:


GoTData.tail(6)


# In[ ]:


GoTData.columns


# Line Plot 

# In[ ]:


GoTData["Book of Death"].plot(kind='line',color = 'purple',label = "Book of Death", linewidth=1,alpha=1,grid=True,linestyle= "-.")
GoTData["Death Chapter"].plot(kind="line",color = "blue", label = "Death Chapter",linewidth=1,alpha=1,grid = True,linestyle = ":")
plt.legend(loc="upper left")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Line Plot Visualization")
plt.show()


# Scatter Plot

# In[ ]:


GoTData.plot(kind = "scatter",x= "Book of Death",y="Death Chapter",alpha=.5,color="black")
plt.xlabel("Death Chapter")
plt.ylabel("Book of Death")
plt.title("Book of Death and Death Chapter 's Scatter Plot")


# Histogram

# In[ ]:


GoTData["Death Chapter"].plot(kind="hist",bins=40,figsize=(10,10))
plt.xlabel("Death Chapter")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


# Cleaning plot
GoTData["Book of Death"].plot(kind="hist",bins=10)
plt.clf()


# In[ ]:


GoTCharacters = GoTData["Name"]


# To check for a character whether in Game of Thrones or not

# In[ ]:


ChName = "Ygritte"
for each in GoTCharacters:
    if each == ChName:
        print(ChName + " is a Game of Thrones Character")


# In[ ]:


print(type(GoTCharacters)) #GoTCharacters column data frame is a series 
print(type(GoTData[["Name"]]))


# Filtering

# In[ ]:


BiC42 = (GoTData["Book Intro Chapter"]<92) & (GoTData["Book Intro Chapter"]>75)
GoTData[BiC42]  
# OR We can use logical_and like this
GoTData[np.logical_and((GoTData["Book Intro Chapter"]>75), (GoTData["Nobility"]==1))]


# In[ ]:


# Looking for characters' rankings
for index,value in enumerate(GoTData["Name"]):
    print(value,"is the",index+1,". character of the Game of Thrones list.")


# In[ ]:


for index,value in GoTData[['Book Intro Chapter']][0:1].iterrows():
    print(index," : ",value)


# In[ ]:





# 

# 
