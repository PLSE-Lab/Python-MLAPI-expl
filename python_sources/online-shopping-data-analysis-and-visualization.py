#!/usr/bin/env python
# coding: utf-8

# ![](http://https://www.google.com/search?q=E+commerce+images&safe=active&rlz=1C1CHBF_enIN831IN831&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiS056D5JjiAhVJYo8KHYHSCjoQ_AUIDigB&biw=1517&bih=640#imgrc=QRU33yRvohrxvM:)

# In[8]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #importing data visualization libraries
import matplotlib.pyplot as mplt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.


# In[2]:


#reading the dataset file
df=pd.read_csv("../input/online_shoppers_intention.csv")


# In[4]:


#VIewing the first 10 rows of the dataset
df.head(10)


# In[5]:


#Viewing the last 10 rows of the dataset
df.tail(10)


# In[9]:


#checking the NA values
df.isna().sum()


# As we can see that there are no NA values in the datset. Now lets check the Null values in the datset

# In[11]:


#checking the Null values in the dataset
df.isnull().sum()


# There are no NULL values is the dataset. Now we are all set for data visualization

# In[12]:


#checking the information of the data
df.info()


# In[17]:


df["Administrative"].value_counts().plot.bar(color="purple",figsize=(10,5))
mplt.title("Administrative plot")
mplt.ylabel("Number of counts or visits")
mplt.show()


# In[18]:


#Administrative_Duration values
df["Administrative_Duration"].value_counts().head(50).plot.bar(figsize=(9,6))
mplt.grid()
mplt.title("Time spent by the user on the page")
mplt.ylabel("Time in seconds")
mplt.show()


# In[22]:


sns.distplot(df["Informational_Duration"],kde=True,norm_hist=True)
mplt.title("Time spent by the user on the website")
mplt.grid()


# In[23]:


sns.countplot(x="Weekend",data=df)
mplt.title("Time spent by the customers on the weekend on the website")
mplt.ylabel("Time in seconds")
mplt.show()


# **The most of the customers are not visiting the website on weekends
# **

# In[25]:


sns.distplot(df["PageValues"],kde=False)
mplt.ylabel("Counts of the Pagevalues")


# In[30]:



sns.lineplot(x="ExitRates",y="Revenue",data=df)


# In[31]:


df["BounceRates"].value_counts().plot.bar(color="green",figsize=(9,5))
mplt.title("Bounce Rates")
mplt.ylabel("Number of Bounce Rates")
mplt.show()


# Revenue is effected directly by Exitrates

# In[34]:



sns.countplot(x="VisitorType",data=df)
mplt.title("Type of visitors on the website")


# In[33]:


sns.lineplot(x="OperatingSystems",y="TrafficType",data=df)
mplt.title("Operating System vs Traffic types")


# **Traffic increases gradually when the number of operating system increase 
# There is a decrease in traffic type when there are 3-4 operating systems and when the operating system inccrease to to 4-6
# **

# In[35]:



sns.lmplot(x="ProductRelated_Duration",y="Weekend",data=df)
mplt.title("Time spent on the products on the weekends")


# In[44]:



sns.scatterplot(x="BounceRates",y="Revenue",data=df)
mplt.title("BounceRates vs Revenue")


# In[46]:


mplt.rcParams['figure.figsize'] = (30, 20)
sns.heatmap(df.corr(),vmin=-2,vmax=2,annot=True)

mplt.title("Heatmap for the Features")


# In[47]:


sns.catplot(x="TrafficType",data=df,alpha=.4)
mplt.title("Type of traffic on the website")


# In[48]:



df.SpecialDay.value_counts().plot.bar(figsize=(9,5))
mplt.title("Effect of special days on the website")


# **The users have low impact of special days **

# In[49]:



sns.countplot(x="Month",data=df)
mplt.title("Monthly visits of the users on teh website")


# The users have visited the website in the month of May and Nov the most
# 

# In[50]:


sns.lmplot(x="Browser",y="TrafficType",data=df)
mplt.title("Different Browser vs Traffic type")


# In[52]:


sns.countplot(x="Browser",data=df)
mplt.title("Number of Browsers used to visit the website")


# In[53]:


label=["2","1","3","4","8","6","7","5"]
size=[6601,2585,2555,478,79,19,7,6]
colors = ['red', 'yellow', 'green', 'pink', 'blue',"orange","purple","black"]


# In[55]:


#pie chart for the Operating systems
mplt.rcParams["figure.figsize"]=(9,9)
mplt.title("website used by different opearting systems")
mplt.pie(size,labels=label,colors=colors,shadow=True,explode = [0.1, 0.1, 0.2, 0.3, .4,.5,.6,.7])
mplt.legend()


# **If you find this kernel helpful kindly upvote this kernel. Suggestions ,edit ,tips are most welcome.**
