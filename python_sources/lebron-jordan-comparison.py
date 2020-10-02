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
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


lebrondata=pd.read_csv("../input/lebron_career.csv")
jordandata=pd.read_csv("../input/jordan_career.csv")


# In[ ]:


lebrondata.info()


# In[ ]:


jordandata.columns


# In[ ]:


lebrondata.describe()


# In[ ]:


jordandata.describe()


# ****LeBron and jordan points means 

# In[ ]:


jordandata.pts.describe()


# In[ ]:


lebrondata.pts.describe()


# asist means for Lebron and Jordan 

# In[ ]:


jordandata.ast.mean()


# In[ ]:


lebrondata.ast.mean()


# Lebron points mean  for first 100 match 

# In[ ]:



lebrondata.head(100).pts.mean()  


# Jordan points mean  for first 100 match

# In[ ]:


#jordan points mean  for first 100 match 
jordandata.head(100).pts.mean()  


# In[ ]:


lebrondata.corr()


# In[ ]:


buyutme=plt.subplots(figsize=(20,20))
sns.heatmap(lebrondata.corr(),annot=True,linewidth=1,fmt='.1f')


# Three points comparison for LeBron vs Jordan

# In[ ]:



lebrondata.three.plot(kind="line",color="yellow",linewidth=1,alpha=1,grid=True,label="LeBron three points",figsize=(10,10))
jordandata.three.plot(kind="line",color="red",linewidth=1,alpha=1,grid=True,label="Jordan three points",figsize=(10,10))
plt.legend(loc='upper right') 
plt.xlabel('match')
plt.ylabel('three points')
plt.title('LeBRON23 VS JORDAN23 for three points') 
plt.show()


# fg-pts scatter plot 

# In[ ]:


lebrondata.plot(kind="scatter",x="fg",y="pts",color="black",figsize=(10,10))
plt.xlabel("fg")
plt.ylabel("pts")
plt.title("fg-pts scatter plot")
plt.show()


# In[ ]:


jordandata.plot(kind="scatter",x="fg",y="pts",color="black",figsize=(10,10))
plt.xlabel("fg")
plt.ylabel("pts")
plt.title("fg-pts scatter plot")
plt.show()


# **Lebron vs jordan for rebound****

# In[ ]:


lebrondata.trb.plot(kind="hist",bins=30,color="black",figsize=(15,15))
jordandata.trb.plot(kind="hist",bins=30,color="red",figsize=(15,15))
plt.xlabel("rebound")
plt.legend(loc='upper right')


# Lebron james comprasion to defensive and offensive rebound 

# In[ ]:



lebrondata.drb.plot(kind="hist",bins=30,color="blue",figsize=(15,15))
lebrondata.orb.plot(kind="hist",bins=30,color="red",figsize=(15,15))
plt.title("James offensive rebound and James defensive rebound histogram  defensive rebound blue -offensive rebound red ")
plt.legend(loc='upper right')
plt.show


# LeBron James 50 points and more 

# In[ ]:


x1=lebrondata["pts"]>=50
lebrondata[x1]


# Micheal Jordan 50 points and more 

# In[ ]:


x2=jordandata["pts"]>=50
jordandata[x2]


# In[ ]:


jordandata[(jordandata["pts"]>35) & (jordandata["ast"]>10) & (jordandata["trb"]>10)]


# In[ ]:


lebrondata[(lebrondata["pts"]>35) & (lebrondata["ast"]>10) & (lebrondata["trb"]>10)]


# Cleveland Cavaliers Lebron vs Miami Heat Lebron for FG

# In[ ]:


lebrondata[(lebrondata["team"]=="CLE") & (lebrondata["fg"]>15)]


# In[ ]:


lebrondata[(lebrondata["team"]=="MIA") & (lebrondata["fg"]>15)]


# In[ ]:


jordandata[jordandata["fg"]>15]


# In[ ]:


lebrondata.fg.plot(kind="line",color="black",alpha=0.5,linewidth=2,figsize=(18,18))
jordandata.fg.plot(kind="line",color="red",alpha=0.5 ,linewidth=2,figsize=(18,18))
plt.xlabel("match")
plt.ylabel("fg")
plt.title("LeBron vs Jordan for fg ")
plt.legend()


# In[ ]:


lebrondata.ast.plot(kind="line",color="black",alpha=0.5,linewidth=2,figsize=(18,18))
jordandata.ast.plot(kind="line",color="red",alpha=0.5 ,linewidth=2,figsize=(18,18))
plt.xlabel("match")
plt.ylabel("asist")
plt.title("LeBron vs Jordan for asist plot Lebron is black Jordan is red")
plt.legend()


# In[ ]:


x3=lebrondata["team"]=="CLE"
lebrondata[x3].pts.mean()


# In[ ]:


# x4=lebrondata["team"]=="MIA"
# lebrondata[x4].pts


# In[ ]:


print(lebrondata.team.value_counts(dropna=False))


# 
# 

# In[ ]:


print(jordandata.pts.value_counts(dropna=False))


# In[ ]:


print(lebrondata.pts.value_counts(dropna=False))


# In[ ]:


lebrondata.boxplot(column="pts",by="ast")


# In[ ]:


jordandata.boxplot(column="pts",by="ast")


# In[ ]:


lebrondata.boxplot(column="pts",by="fg")


# In[ ]:


jordandata.boxplot(column="pts",by="fg")


# In[ ]:


lebrondata_new=lebrondata.head()


# In[ ]:





# In[ ]:




