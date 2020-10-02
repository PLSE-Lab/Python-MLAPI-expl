#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Data manipulation modules
import pandas as pd        # R-like data manipulation
import numpy as np         # n-dimensional arrays - Base Package

#For pltting
import matplotlib.pyplot as plt      # For base plotting
# Seaborn is a library for making statistical graphics
# in Python. It is built on top of matplotlib and 
#  numpy and pandas data structures.
import seaborn as sns                # Easier plotting

#Misc - Will use for reading data from file
import os


# In[ ]:


#Set working directory
os.chdir("/Users/digvijay/Desktop/study/BigData/kaggle/GunViolence")
os.listdir()


# In[ ]:


gunD = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")


# In[ ]:


type(gunD)


# In[ ]:


gunD.shape
gunD.info()


# In[ ]:


gunD.columns


# In[ ]:


gunD.head(10)


# In[ ]:


gunD['state'].unique()


# In[ ]:


grouped = gunD.groupby('state').size().reset_index(name='counts').sort_values(by=('counts'),ascending=False)


# In[ ]:


grouped.counts = grouped.counts.astype(int)


# In[ ]:


grouped.info()


# In[ ]:


plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="state", y="counts", data=grouped)

plot1.set_xticklabels(grouped['state'], rotation=90, ha="center")
plot1.set(xlabel='States',ylabel='Counts')
plot1.set_title('Gun violance incidents recorded state wise')
plt.show()


# In[ ]:


plt.figure(figsize = (20,20))
labels = grouped.state
plt.pie(grouped.counts, autopct='%1.1f%%')
plt.legend(labels, loc="best")
plt.title('Goal Score', fontsize = 20)
plt.axis('equal')
plt.show()


# In[ ]:


gunD['date'] = pd.to_datetime(gunD['date'])


# In[ ]:


gunD['month'] = gunD.date.map(lambda date: date.month)


# In[ ]:


gunD['year'] = gunD.date.map(lambda date: date.year)


# In[ ]:


gunD.head(10)


# In[ ]:


grouped = gunD.groupby('state')
g=grouped['n_killed'].agg([np.sum]).reset_index().sort_values(by=('sum'),ascending=False)


# In[ ]:


g=pd.DataFrame(g)
g


# In[ ]:


g.info()
g.shape


# In[ ]:


plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot2 = sns.barplot(x="state", y="sum", data=g)

plot2.set_xticklabels(g['state'], rotation=90, ha="center")
plot2.set(xlabel='States',ylabel='sum')
plot2.set_title('People killed state wise')
plt.show()


# In[ ]:


grouped = gunD.groupby('state')
g=grouped['n_injured'].agg([np.sum]).reset_index().sort_values(by=('sum'),ascending=False)


# In[ ]:


plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot3 = sns.barplot(x="state", y="sum", data=g)

plot3.set_xticklabels(g['state'], rotation=90, ha="center")
plot3.set(xlabel='States',ylabel='sum')
plot3.set_title('People injured state wise')
plt.show()


# In[ ]:


grouped = gunD.groupby('state')
g=grouped['n_injured','n_killed'].agg([np.sum]).reset_index()


# In[ ]:


g.shape


# In[ ]:


g.head()


# In[ ]:


g.n_injured.shape


# In[ ]:


g.plot(x="state", y=["n_injured", "n_killed"], kind="bar")


# In[ ]:


plt.figure(figsize = (20, 20), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.0)
plot3 = g.plot(x="state", y=["n_injured", "n_killed"], kind="bar")

plot3.set_xticklabels(g['state'], rotation=90, ha="center")
plot3.set(xlabel='States',ylabel='sum')
plot3.set_title('People injured and killed state wise')
plt.show()


# In[ ]:


grouped = gunD.groupby(['state','city_or_county']).size().reset_index(name='counts').sort_values(by=('counts'),ascending=False)


# In[ ]:


g=grouped.loc[grouped.state=="Illinois",:]


# In[ ]:


g=pd.DataFrame(g)


# In[ ]:


g.columns


# In[ ]:


g = g.loc[g.counts>10,['city_or_county','counts']]


# In[ ]:


g.head(30)


# In[ ]:


plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot3 = sns.barplot(x="city_or_county", y="counts", data=g)

plot3.set_xticklabels(g['city_or_county'], rotation=90, ha="center")
plot3.set(xlabel='City or conuty',ylabel='counts')
plot3.set_title('Crimes reported in City/County of Illinois')
plt.show()


# In[ ]:


plt.figure(figsize = (10,10))
labels = g.city_or_county
plt.pie(g.counts, autopct='%1.1f%%')
plt.legend(labels, loc="best")
plt.title('Goal Score', fontsize = 20)
plt.axis('equal')
plt.show()


# In[ ]:




