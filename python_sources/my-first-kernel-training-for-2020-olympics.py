#!/usr/bin/env python
# coding: utf-8

# This kernel is the very first step for my data science studies.
# I tried to analyze and visualize the daha which i found in dataset named  "120 years of Olympic history: athletes and results".
# I will upgrade this kernel with my progress in Udemy Class from DATAI Team

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  #  visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# **IMPORTING DATA**

# In[ ]:


data = pd.read_csv("../input/athlete_events.csv")
data2 = pd.read_csv("../input/noc_regions.csv")


# **TAKING INFORMATION FROM DATA**

# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


data.describe()


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= ".1f",ax=ax)
plt.show()


# In[ ]:


data.columns


# In[ ]:


data["Name"]


# In[ ]:


print(data["Team"].value_counts(dropna=False))


# **CONTROLLING DATA WITH HEAD AND TAIL CODES**

# In[ ]:


data.head(10)


# In[ ]:


data.tail(10)


# **SEX**

# In[ ]:


print("Male:",len(data[data.Sex == "M"]))
print("Female:",len(data[data.Sex == "F"]))


# **PLOTS**

# In[ ]:


# bar
y = np.array([len(data[data.Sex == "M"]),len(data[data.Sex == "F"])])
x = ["Male","Female"]

plt.bar(x,y)
plt.xlabel("Sex")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


#Line Plot
data.Height.plot(kind = "line", color = "blue", label = "Height", linewidth = 0.5, alpha = 0.5,grid = True,linestyle = "-.")
data.Weight.plot(kind = "line", color = "red" , label = "Weight", linewidth = 0.5, alpha = 0.5,grid = True,linestyle = "-" )
plt.legend(loc = "upper left")
plt.xlabel("index")
plt.ylabel("value")
plt.title("Weight-Height for athletes")
plt.show()


# In[ ]:


#Scatter Plot
data.plot(kind = "scatter", x = "Weight", y = "Height", alpha = 0.1, color = "blue")
plt.xlabel("Weight")
plt.ylabel("Height")
plt.title("Weight-Height Scatter Plot")


# In[ ]:


#Histogram
data.Age.plot(kind = "hist",bins = 100, figsize =(12,12) )
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# **AGE**

# In[ ]:


data.Age.plot(kind = "hist",bins = 50, figsize =(12,12) )
plt.clf()


# DICTIONARY and PANDAS EXAMPLES

# In[ ]:


dictionary = {"John_Aalberg" : "UnitedStates", "Antti_Sami_Aalto" : "Finland"}
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


dictionary["John_Aalberg"] = "USA"
print(dictionary)
dictionary["Pepijn_Aardewi_jn"] = "Netherlands"
print(dictionary)
del dictionary["John_Aalberg"]
print(dictionary)
print("John_Aalberg" in dictionary) 
dictionary.clear()
print(dictionary)


# In[ ]:


series = data["Weight"]
print(type(series))
data_frame = data[["Weight"]]
print(type(data_frame))


# In[ ]:


data.loc[:10, "Name"]  #"pandas.core.series.Series"


# In[ ]:


data.loc[:10, ["Name"]] #"pandas.core.frame.DataFrame"


# In[ ]:


data.loc[0:100:10, "Name":"Team"]


# In[ ]:


data.loc[100:0:-10, "Name":"Team"]


# In[ ]:


data["Qualification"] = ["Nice" if each == "Gold" else "Bad" for each in data.Medal]
data.head()


# In[ ]:


# example
pi = 3.14
print(pi > 3)
print(pi!=3.14)
# Boolean operators
print(True & False)
print(True or False)


# In[ ]:


x = data["Age"] > 80
data[x]


# In[ ]:


data[(data["Age"]>80) & (data["Year"]<1935)]


# In[ ]:


i = 0
while i != 20 :
    print('i is: ',i)
    i +=4 
print(i,' is equal to 20')


# In[ ]:


lis = [33,55,11,22,66]
for each in lis:
    print('each is: ',each)
print('')

# Enumerate index and value of list
# index : value = 0:33, 1:55, 2:11, 3:22, 4:66
for index, value in enumerate(lis):
    print(index," : ",value)
print('')   


# For pandas we can achieve index and value
for index,value in data[['Year']][0:1].iterrows():
    print(index," : ",value)

