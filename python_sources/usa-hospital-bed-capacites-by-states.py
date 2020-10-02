#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

import seaborn as sns # visualization library
import plotly as py # visualization library
from plotly.offline import init_notebook_mode, iplot # plotly offline mode
init_notebook_mode(connected=True) 
import plotly.graph_objs as go # plotly graphical object

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


USA=pd.read_csv("../input/global-hospital-beds-capacity-for-covid19/hospital_beds_USA_v1.csv")


# In[ ]:


#Sorting values firstly Country then Year ascending order
USA.sort_values(['state', 'year'], ascending=[True, True], inplace=True)


# In[ ]:


USA.columns=map(lambda x:str(x).upper(), USA.columns)


# In[ ]:


USA.drop(USA.iloc[:,-2:], inplace = True, axis = 1)


# In[ ]:


global_=pd.read_csv("../input/global-hospital-beds-capacity-for-covid19/hospital_beds_global_regional_v1.csv")
global_.sort_values(['country', 'year'], ascending=[True, True], inplace=True)
global_.columns=map(lambda x:str(x).upper(), global_.columns)
global_.drop(global_.iloc[:,-2:], inplace = True, axis = 1)
global_.head()


# In[ ]:


global_.info()


# In[ ]:


global_.describe().T


# In[ ]:


list(set(global_.dtypes.tolist()))


# In[ ]:


df_num = global_.select_dtypes(include = ['float64', 'int64'])
df_num.head()


# In[ ]:


global_[['BEDS','POPULATION']].corr() 


# Frankly, I would expect the population size and bed capacities to be highly correlated. 

# In[ ]:


#USA's population and bed capacity correlation by years and patient service type
sns.lmplot(x="POPULATION", y="BEDS",hue="YEAR",data=USA ,palette="Set2");


# **DETAIL ANALYSIS OF USA AND GLOBAL DATA**

# In[ ]:


USA.isnull().sum()


# In[ ]:


from collections import Counter

counter = Counter(USA['STATE'])
# Printing biggest values
counter.most_common(10)


# In[ ]:


group1=USA.groupby(['YEAR'])['POPULATION','BEDS'].sum()
group_1=group1.reset_index()
group_1=group_1.iloc[ 0:8 , : ]   #I did not include it in the table due to the low number of 2020.
group_1   


# In[ ]:


f, axes = plt.subplots(1, 2,figsize=(15,8))
sns.lineplot(  y="POPULATION", x= "YEAR", data=group_1 , ax=axes[0])
sns.lineplot(  y="BEDS", x= "YEAR", data=group_1,  ax=axes[1]);


# In[ ]:


group2=USA.groupby(['STATE','YEAR'])['POPULATION','BEDS'].sum()
group_2=group2.reset_index()


# In[ ]:


# Get names of indexes for which column Year has value 2020
indexNames = group_2[ group_2['YEAR'] == 2020 ].index
# Delete these row indexes from dataFrame
group_2.drop(indexNames , inplace=True)


# In[ ]:


# A few states from the USA
group2_MO= group_2[group_2['STATE'] == "MO"]
group2_LA= group_2[group_2['STATE'] == "LA"]
group2_MI= group_2[group_2['STATE'] == "MI"]
group2_WA= group_2[group_2['STATE'] == "WA"]
group2_TX= group_2[group_2['STATE'] == "TX"]
group2_CA= group_2[group_2['STATE'] == "CA"]
group2_FL= group_2[group_2['STATE'] == "FL"]


# In[ ]:


with plt.xkcd():
    plt.plot(group2_MO.YEAR,group2_MO.BEDS, linestyle="--",linewidth=3,label = "MO")
    plt.plot(group2_LA.YEAR,group2_LA.BEDS, color="#20B2AA",linewidth=3,label= "LA")
    plt.plot(group2_MI.YEAR,group2_MI.BEDS,color="#A153E1",linewidth=3, label="MI")
    plt.plot(group2_WA.YEAR,group2_WA.BEDS,color="#1FC9A5",linewidth=3, label="WA")
    plt.plot(group2_TX.YEAR,group2_TX.BEDS,color="#E9967A",linewidth=3, label="TX")
    plt.plot(group2_CA.YEAR,group2_CA.BEDS,color="#FFD700",linewidth=3, label="CA")
    plt.plot(group2_FL.YEAR,group2_FL.BEDS,color="#9ACD32",linewidth=3, label="FL")

    
    plt.xlabel("Year")
    plt.ylabel("Bed Capacity")
    plt.legend()
    plt.tight_layout()
    plt.title("Bed Capacities by States in USA")
    plt.show()


# In[ ]:


with plt.xkcd():

   plt.plot(group2_MO.YEAR,group2_MO.POPULATION, linestyle="--",linewidth=3,label = "MO")
   plt.plot(group2_LA.YEAR,group2_LA.POPULATION, color="#20B2AA",linewidth=3,label= "LA")
   plt.plot(group2_MI.YEAR,group2_MI.POPULATION,color="#A153E1",linewidth=3, label="MI")
   plt.plot(group2_WA.YEAR,group2_WA.POPULATION,color="#1FC9A5",linewidth=3, label="WA")
   plt.plot(group2_TX.YEAR,group2_TX.POPULATION,color="#E9967A",linewidth=3, label="TX")
   plt.plot(group2_CA.YEAR,group2_CA.POPULATION,color="#FFD700",linewidth=3, label="CA")
   plt.plot(group2_FL.YEAR,group2_FL.POPULATION,color="#9ACD32",linewidth=3, label="FL")

   
   plt.xlabel("Year")
   plt.ylabel("Population")
   plt.legend()
   plt.tight_layout()
   plt.title("Population by States in USA")
   plt.show()   


# In[ ]:


group3=USA.groupby('STATE')['POPULATION','BEDS'].mean()
group_3=group3.reset_index()

group_4=group_3.sort_values(by=['POPULATION'], ascending=False)
group_5=group_4.iloc[:11,:]  # Top 10 states according to State.


# In[ ]:


f, axes = plt.subplots(1, 2,figsize=(15,8))
sns.barplot(  y="POPULATION", x= "STATE", data=group_5 , ax=axes[0])
sns.barplot(  y="BEDS", x= "STATE", data=group_5,  ax=axes[1]);


# In[ ]:


a=USA.groupby(['TYPE','YEAR'])['BEDS'].mean()
b=a.reset_index()


# In[ ]:


plt.figure(figsize = (10, 6))
ax = sns.boxplot(x='TYPE', y='BEDS', data=b)
ax.set_title('Bed capacities according to types in USA')
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45);


# In[ ]:


import plotly.express as px
fig = px.bar(b[['YEAR', 'BEDS','TYPE']].sort_values('BEDS', ascending=False), 
             y="BEDS", x="YEAR", color='TYPE', 
             log_y=True, template='ggplot2')
fig.update_layout(title_text='Beds capacities according to years in USA')
fig.show()


# In[ ]:


USA_ACUTE= b[b['TYPE'] == "ACUTE"]
USA_ICU=b[b['TYPE'] == "ICU"]
USA_OTHER=b[b['TYPE'] == "OTHER"]
USA_PSYCHIATRIC=b[b['TYPE'] == "PSYCHIATRIC"]


# In[ ]:


with plt.xkcd():
    plt.figure(figsize=(12,8))
    plt.plot(USA_ACUTE.YEAR,USA_ACUTE.BEDS, linestyle="--",linewidth=2,label = "Acute")
    plt.plot(USA_ICU.YEAR,USA_ICU.BEDS, color="#4EB772",linewidth=2,label= "ICU")
    plt.plot(USA_OTHER.YEAR,USA_OTHER.BEDS,color="#A153E1",linewidth=2, label="Other")
    plt.plot(USA_PSYCHIATRIC.YEAR,USA_PSYCHIATRIC.BEDS,color="#1FC9A5",linewidth=2, label="Psychiatric")
    
    plt.xlabel("Year")
    plt.ylabel("Bed Capacity")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)
    plt.tight_layout()
    plt.title("Bed Capacities by Types in USA")
    plt.show()


# * As you can see, the bed capacities generally decrease over the years in the USA.(Especially psychiatric and other type.)
# * Increase is observed except for ICU (Intensive Care Unit) 2019 and 2020.

# In[ ]:


c=global_.groupby(['TYPE','YEAR'])['BEDS'].mean()
d=c.reset_index()


# In[ ]:


# Get names of indexes for which column Year has value less than 2011
indexNames2 = d[ d['YEAR'] < 2011 ].index
# Delete these row indexes from dataFrame
d.drop(indexNames2 , inplace=True)


# In[ ]:


plt.figure(figsize = (10, 6))
ax = sns.boxplot(x='YEAR', y='BEDS', data=d)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('BEDS')
plt.title('Box Plot-Global Data');


# In[ ]:


from collections import Counter
counter = Counter(global_['TYPE'])
counter


# In[ ]:


global_ACUTE= d[d['TYPE'] == "ACUTE"]
global_ICU=d[d['TYPE'] == "ICU"]
global_OTHER=d[d['TYPE'] == "OTHER"]
global_PSYCHIATRIC=d[d['TYPE'] == "PSYCHIATRIC"]
global_TOTAL=d[d['TYPE'] == "TOTAL" ]


# In[ ]:


with plt.xkcd():
    plt.figure(figsize=(12,8))
    plt.plot(global_ACUTE.YEAR,global_ACUTE.BEDS,color="#355C7D", linestyle="--",linewidth=3,label = "Acute")
    plt.plot(global_ICU.YEAR,global_ICU.BEDS, color="#F8B195",linewidth=3,label= "ICU")
    plt.plot(global_OTHER.YEAR,global_OTHER.BEDS,color="#F67280",linewidth=3, label="Other")
    plt.plot(global_PSYCHIATRIC.YEAR,global_PSYCHIATRIC.BEDS,color="#C06C84",linewidth=3, label="Psychiatric")
    plt.plot(global_TOTAL.YEAR,global_TOTAL.BEDS,color="#6C5B7B",linewidth=3, label="Total")
    
    plt.xlabel("Year")
    plt.ylabel("Bed Capacity")
    plt.legend()
    plt.tight_layout()
    plt.title("Bed capacities by types in global")
    plt.show()


# * The bed capacity increase in psychiatric services bed capacity in 2019 and total services bed capacity in 2017 is remarkable.

# In[ ]:


pop_group = USA.groupby(["STATE","COUNTY"])[["POPULATION","BEDS"]].mean().reset_index()
pop_group.head()


# In[ ]:


pop_group2 = pop_group.sort_values(by=['STATE','POPULATION'],ascending=False)
pop_group2=pop_group2.groupby('STATE').head(5) # first 5 rows for each group


# In[ ]:


fig = px.bar(pop_group2[['STATE', 'POPULATION','COUNTY']].sort_values('POPULATION', ascending=False), 
             y="POPULATION", x="STATE", color='COUNTY', 
             log_y=True, template='ggplot2')
fig.show()

