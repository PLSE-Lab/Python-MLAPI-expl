#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


mafs=pd.read_csv(os.path.join(dirname,filename))
mafs


# In[ ]:


#importing necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#lets get shape and size and more
print(mafs.shape)
print(mafs.ndim)
print(mafs.size)


# In[ ]:


#lets know unique values of the tables.

def uniqueincsv(data):
    for i in data:
        print(i,"=",data[i].unique(),end="\n\n")


# In[ ]:


uniqueincsv(mafs)


# In[ ]:


#lets make a deep copy first
mafs1=mafs.copy(deep=True)


# In[ ]:


#lets see the number of participants in each season using a histogram
plt.figure()
g=sns.distplot(mafs1["Season"], kde=False, bins=10, hist_kws=dict(edgecolor="k", linewidth=2))
g.set_title("Number of Seasons with Number of Participants")
plt.xlim([1,10])


# In[ ]:


#Lets see the age variation using a histogram
plt.figure()
A=plt.hist(mafs1["Age"], edgecolor="red")


# In[ ]:


#lets see in which city, a particular season was shot.
#We will achieve this by making a bar chat
plt.figure(figsize=(16,4))
m=plt.bar(mafs1["Location"], mafs1["Season"])
plt.xticks(rotation=90)
plt.xlabel("City")
plt.ylabel("Seasons")


# In[ ]:


#Lets print nuber of participant from each city
a=plt.hist(mafs1["Location"], edgecolor="yellow")
plt.xticks(rotation=90)


# In[ ]:


#Decision leads to status, if both Male and Female agrees, the they stay married.
#If either disagrees, then both get divorced
#Let us see how many males and females said yes or no. 
#I am gonna convert decision column into dummy
dummydes=pd.get_dummies(mafs1["Decision"])
dummydes


# In[ ]:


mafs1=pd.concat([mafs1,dummydes], axis=1)
mafs1.drop(["No","Decision"], inplace=True, axis=1)
mafs1


# In[ ]:


#Yes=1 and No=0, I have removed Decision column as well as No column
#Now lets see how many people said 1 or 0
plt.hist(mafs1["Yes"], color="green")
print(mafs1["Yes"].value_counts())


# In[ ]:


#A better represestation using pie chart
labels="Yes","NO"
sizes=mafs1["Yes"].value_counts()
explode=(0,0.2)
fig, ax=plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)
ax.axis('equal')
plt.show()


# In[ ]:


#showing pie chart for % of M and female participant
lables="Female","Male"
sizes=mafs1["Gender"].value_counts()
explode=(0,0)
fig, ax=plt.subplots()
ax.pie(sizes, explode=explode, labels=lables, autopct="%1.1f%%", shadow=True, startangle=90)
ax.axis('equal')
plt.show()


# In[ ]:


maleyes=0
maleno=0
femaleyes=0
femaleno=0
for i in mafs1.index:
    if mafs1["Gender"][i]=="M" and mafs1["Yes"][i]==1:
        maleyes+=1
    elif mafs1["Gender"][i]=="M" and mafs1["Yes"][i]==0:
        maleno+=1
    elif mafs1["Gender"][i]=="F" and mafs1["Yes"][i]==1:
        femaleyes+=1
    elif mafs1["Gender"][i]=="F" and mafs1["Yes"][i]==0:
        femaleno+=1
    else:
        continue
        
print("{0} {1} {2} {3}".format(maleyes, maleno, femaleyes, femaleno))


# In[ ]:


#pie chart representation of males and females saying yes and no respectively
lables="MY","MN", "FY", "FN"
sizes=[maleyes,maleno,femaleyes,femaleno]
explode=(0.09,0.09,0.09,0.09)
fig, ax=plt.subplots()
ax.pie(sizes, explode=explode, labels=lables, autopct="%1.1f%%", shadow=True, startangle=90)
ax.axis('equal')
plt.show()


# In[ ]:


#Now lets see how many married couple and divorced couple are there
#We will also do this by using a pie chart
div,mar=mafs1["Status"].value_counts()


# In[ ]:


mar, div


# In[ ]:


#showing pie chart for % of M and female participant
lables="Married","Divorced"
sizes=[mar,div]
explode=(0.1,0)
fig, ax=plt.subplots()
ax.pie(sizes, explode=explode, labels=lables, autopct="%1.1f%%", shadow=True, startangle=90)
ax.axis('equal')
plt.show()


# In[ ]:




