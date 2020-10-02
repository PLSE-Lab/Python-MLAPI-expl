#!/usr/bin/env python
# coding: utf-8

# **So The following kernal gives a brief analysis of air-crashes globally.
# The dataset dates back to the 1900s which holds the details of aircraft crash.**
# 
# **In this kernel i have mostly focussed on Airbus and Boeing aircraft's as these are operated globally as commercial civillian airlines. And thus carried out the analysis like the major site of crashes, year-wise crash etc. which you are going to find the following subsections**
# 
# Feel free to ask anything in comment section and suggestions are welcomed too.
# 

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


#   **Let's load the dataset and inspect a first few 5 entires** 

# In[ ]:


df=pd.read_csv("../input/planecrashinfo_20181121001952.csv",na_values='?')
#df.info()
l1=df.summary.dropna().map(lambda x: True if "midair" in x else False  )
l1.value_counts()
df[df.operator.str.contains("National")==True]
#df.fatalities
df.count()
df[df.ac_type.str.contains("737")==True].count()

df.head(5)


# **Extracting only the fatality and aboard count and converting it to type int.
# The following modification is done as I'm interested only in the no. of fatalitites and thus being able to convert it into int. for furthur analysis. **

# In[ ]:


import matplotlib.pyplot as plt
df.groupby(['fatalities','route']).fatalities.count()



#cleaning the fatalitites column
pc=df.copy()

pc['fatalities']=np.where(df['fatalities'].str[0]=='?',0, df['fatalities'].str[0:4])
pc['aboard']=np.where(df['aboard'].str[0]=='?',0, df['aboard'].str[0:4])
pc.fatalities.astype("str")
pc.aboard.astype("str")
pc['fatalities']=pc['fatalities'].str.strip()
pc['aboard']=pc['aboard'].str.strip()


#print(pc.fatalities[101])
pc['fatalities']=pd.to_numeric(pc['fatalities'])
pc['aboard']=pd.to_numeric(pc['aboard'])

pc.info()


pc.groupby('fatalities').fatalities.count()

pc[pc['fatalities']<100]['fatalities'].plot.hist()
pc.head(5)


# **converting the DD-MM-YYYY column to just YYYY. This modification is same as the one done above. So as to classify the crashes w.r.t. to year.**
# 

# In[ ]:


pc.describe(include="all")
pc['date']=pc['date'].str[-4:]
pc['date']=pd.to_numeric(pc['date'])
pc['date']
pc.head()

#pc.groupby('date').date.count()
#pc.query("date == 2017")


# > **the following hexplot shows the fatality count wrt to years**

# In[ ]:


pc.plot.hexbin(x='fatalities', y='date' ,gridsize=15)


# In[ ]:


pc['date']=pd.to_numeric(pc['date'])
pc[pc['fatalities'] < 100].plot.hexbin(x='fatalities', y='date',gridsize=15)


# **total no. of fatalities from 2010 onwards**

# In[ ]:


pc.head(10)
pc[pc['date']>2010].groupby('date').fatalities.sum().plot(kind='bar')


# In[ ]:


#pc.groupby('route').fatalities.sum().plot(kind='bar')
#pc.groupby('operator').fatalities.sum().max()
pc.groupby('ac_type').fatalities.sum()
#pc['route']
a=np.where(pc.date>2000)
#print (a)


# **In this next section I've created a seperate feature - 'company' and narrowing the dataset by only analyzing boeing and airbus's aircrafts. As these are dominant in civil airlines
# 
# And then calculating the fatalities.
# I've chosen data > 1970 so as to have fair analysis. As airbus came into service in 1970 unlike boeing which were operational in early 1930s**
# 

# In[ ]:


a=[1,23,3]
b=[4,5,6]
q=zip(a,b)


# In[ ]:


airb = pc.copy()
#airb.info()
#airb[airb['ac_type'].str.contains("Airbus")==True]
#t=airb.ac_type.notnull()

l=airb[(airb['ac_type'].str.contains("Airbus")==True) | (airb['ac_type'].str.contains("Boeing")==True)]

l['company']=np.where(l['ac_type'].str.contains("Boeing"),'Boeing','Airbus')

l.company.value_counts().plot(kind="bar")

boeing=l.company.value_counts()[0]
airbus=l.company.value_counts()[1]


fig = plt.figure(figsize=(15,3))

ax1 = fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
aboard=[
            l[(l['date'] > 1970) & (l['company']=="Boeing")].aboard.sum(),
            l[(l['date'] > 1970) & (l['company']=="Airbus")].aboard.sum()
       ]
fatality=[
             l[(l['date'] > 1970) & (l['company']=="Boeing")].fatalities.sum(),
             l[(l['date'] > 1970) & (l['company']=="Airbus")].fatalities.sum()
         ]

ax1.bar(['boeing','airbus'],
       aboard,
       color='brown')
ax1.bar(['boeing','airbus'],
        fatality,
       
       color='green')
print(aboard, fatality)

ax2.bar(['boeing','airbus'],
       mod,
       color='orange')
#l.head(10)


# **ANow in this section, I want to know how boeing and airbus aircrafts crashed. Basically the cause of crash.
# So the information was extracted from the summary col. by identifying the root cause. For eg. if a plane crashed because of a mountain then all then summary cols which contains 'mountain' word results crash because of collision with mountain. Like wise crash during landing , take off, so on and so forth.**

# In[ ]:


l['model']=l['ac_type'].str[6:]
crash=l.summary.dropna()

mt=crash.map(lambda x: True if ("mountain" in x)   else False  )
sea=crash.map(lambda x: True if ("sea" in x) or ("ocean" in x) else False  )
midair=crash.map(lambda x: True if "midair" in x else False  )
takeoff=crash.map(lambda x: True if ( "taking off" in x) or ("take off" in x) else False  )
land=crash.map(lambda x: True if ("land" in x) or ("landing" in x) else False  )
shotd=crash.map(lambda x: True if "shot down" in x else False  )
unk=crash.map(lambda x: True if "unknown" in x else False  )
fog=crash.map(lambda x: True if ("foggy" in x) or ("fog" in x) else False  )
hij=crash.map(lambda x: True if ("hijacked" in x) or ("hijack" in x) else False  )
birdi=crash.map(lambda x: True if "bird" in x else False  )

print(birdi.value_counts()[1])
#print(hij)
#crash[260]

a=[mt.value_counts()[1],
   sea.value_counts()[1],
   midair.value_counts()[1],
  takeoff.value_counts()[1],
  land.value_counts()[1],
  shotd.value_counts()[1],
  
   fog.value_counts()[1],
   hij.value_counts()[1]
  ]
pd.Series(a, index=["mountain",
                    "sea",
                    "mid-air",
                    "take-off",
                    "landing",
                    "shot down",
                    "fog",
                    "hijacked"
                   ]).plot(kind="bar")



# **In the next section i was trying to find the crashes in a particular ocean (pacific, indian and atlantic) using regex. (Incomplete)**

# In[ ]:


#pc[pc.location.dropna.str.contains("Ocean")]
pc['lcation']=pc.location.dropna()
pc[pc['lcation'].isnull()==True]
#dfc['Time_of_Sail'] = pd.to_datetime(dfc['Time_of_Sail'],format= '%H:%M:%S' ).dt.time
l.head(3)

#l['time']=pd.to_datetime(l['time'],format='%H:%M').dt.time


# **Below model shows the plane-crash involving Boeing 737  and Airbus A320 family. Seems A320 is safer. My assumption :P. **

# In[ ]:


A320=l[l.company.str.contains("Airbus")].model.str.contains("A320").value_counts()[1]
#l[l.company.str.contains("Boeing")].model.str.contains("777").value_counts()
B737=l[l.company.str.contains("Boeing")].model.str.contains("73").value_counts()[1]
typ=[A320,B737]
pd.Series(typ, index=['A320','B737'])


# T**he following plot shows crashes involving ATR aircrafts. We can't deduce any usefull info. But It can be said that minimum 1 carsh per year occurs in ATR**

# In[ ]:


ATR=pc[pc.ac_type.str.contains("ATR")==True]
ATR.groupby("date").date.value_counts().plot(kind="bar",  fontsize =12,
     
      colormap ="winter_r")
#ATR


# **Top 5 airlines with max. no. of crashes.**

# In[ ]:


a=df.operator.value_counts().sort_values(ascending=False).head(5)
type(a)
a
a.plot(kind="bar",
       title="max. crashes", 
       grid=True, fontsize =12,
       legend =True,
      colormap ="rainbow")


# T**With the help of the below line we can extract the count of crashes involved in a particular regional airline. For e.g. in my case I have filtered out the airlines having 'India' as the substring which eventually gives me the airlines of India (The national flag carrier). Which shows that Indian airlines have suffered the most.**

# In[ ]:


df[df.operator.str.contains("India")==True].operator.value_counts()


# In[ ]:


df[df.operator.str.contains("China")==True].operator.value_counts()


# In[ ]:





# In[ ]:




