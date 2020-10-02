#!/usr/bin/env python
# coding: utf-8

# 1. **INTRODUCTION**
# > *The dataset used in this notebook is an open-source database including information on terrorist attacks around the world from 1970 through 2017. This dataset includes systematic data on domestic as well as international terrorist incidents that have occurred during this time period*

# In[ ]:


#importing important libraries
import numpy as np
import csv


# **Number of attacks held between day 10 and day 20**

# In[ ]:


file_obj=open('../input/terrorismData.csv', encoding='utf8')
file_data=csv.DictReader(file_obj, skipinitialspace=True)
days=[]
for row in file_data:
    days.append(row['Day'])
np_days=np.array(days, dtype=float)
s=np_days[(np_days>=10) & (np_days<=20)]
print(s.shape[0])


# **Number of attacks held between 1 Jan 2010 and 31 Jan 2010(including both date).**
# > note: I have ignored the case where day=0, and printed the count of number of attacks as integer value.
# 

# In[ ]:


with open('../input/terrorismData.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    
    day=[]
    year=[]
    month=[]
    count=0
    for row in file_data:
        day.append(row['Day'])
        month.append(row['Month'])
        year.append(row['Year'])
    np_day=np.array(day, dtype='int')
    np_year=np.array(year, dtype='int')
    np_month=np.array(month, dtype='int')
    
    np_day=np_day[np_month==1]
    np_year=np_year[np_month==1]
    np_day=np_day[np_year==2010]
    for i in np_day:
        if i>=1 and i<=31:
            count+=1
    print(count)


# *As we knew the Kargil War that took place between May 1999 and July 1999 (3 Months) ,so there was a huge conflict in Kashmir Valley during this period.*
# > In this dataset, there is no information regarding the war between the two countries to find out the casualty during the war.
# > So I have found out the attack in this period in which maximum casualties happened. and printted the count of casualties (as integer), city in which that attack happened and name of attack group.
# > Where,
# > Casualty = Killed + Wounded.I have filled the empty value in killed or wounded feature to 0.

# In[ ]:


with open("../input/terrorismData.csv") as file:
    file_obj = csv.DictReader(file,skipinitialspace = False)
    killed = []
    wounded = []
    month = []
    year = []
    city = []
    group = []
    
    for row in file_obj:
        if "1999" in row["Year"]:
            if "5" in row["Month"] or  "6" in row["Month"] or "7" in row["Month"]:
                if "Unknown" not in row["City"]:
                    if "Unknown" not in row["Group"]:
                        if "Jammu and Kashmir" in row["State"]:
                            killed.append(row["Killed"])
                            wounded.append(row["Wounded"])
                            city.append(row["City"])
                            group.append(row["Group"])
        
    killed = np.array(killed)
    wounded = np.array(wounded)
    city = np.array(city)
    group = np.array(group)
    
    killed_bool = killed == ""
    wounded_bool = wounded == ""
    
    killed[killed_bool] = "0.0"
    wounded[wounded_bool] = "0.0"
    
    killed = np.array(killed, dtype = float)
    wounded = np.array(wounded, dtype = float)
    
    casualty = (killed + wounded)
    max_casualty = (int)(casualty.max())
    max_casualty_arg = casualty.argmax()
    print(max_casualty,city[max_casualty_arg],group[max_casualty_arg])
    


# Following code gives casualty in the Red Corridor States. Mainly Red corridor states include Jharkhand, Odisha, Andhra Pradesh, and Chhattisgarh.
# Casualty=Killed +Wounded.
# I have printed the count of Casualty as integer value.

# In[ ]:


with open('../input/terrorismData.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    killed=[]
    wounded=[]
    state=[]
    count=0
    for row in file_data:
        killed.append(row['Killed'])
        wounded.append(row['Wounded'])
        state.append(row['State'])
    np_killed=np.array(killed)
    np_killed[np_killed=='']='0'
    np_killed=np.array(np_killed, dtype='float')
    np_wounded=np.array(wounded)
    np_wounded[np_wounded=='']='0'
    np_wounded=np.array(np_wounded, dtype='float')
    np_casuality=np.array(np_killed+np_wounded, dtype='int')
    for i in range(len(state)):
        if state[i]=='Chhattisgarh' or state[i]=='Odisha' or state[i]=='Jharkhand' or state[i]=='Andhra Pradesh':
            count+=np_casuality[i]
    print(count)


# **Top 5 Indian Cities which has most number of casualties**
# > I have printed top 5 cities along with total casualties in that city.

# In[ ]:


with open('../input/terrorismData.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    killed=[]
    wounded=[]
    city=[]
    for row in file_data:
        if 'India' in row['Country'] and 'Unknown' not in row['City']:
            city.append(row['City'])
            wounded.append(row['Wounded'])
            killed.append(row['Killed'])
    np_wounded=np.array(wounded)
    np_killed=np.array(killed)
    np_city=np.array(city)
    
    np_killed[np_killed=='']='0.0'
    np_wounded[np_wounded=='']='0.0'
    np_killed=np.array(np_killed, dtype='float')
    np_wounded=np.array(np_wounded, dtype='float')
    np_casuality=np.array(np_wounded+np_killed, dtype='int')
    
    
    citydic={}
    for i in range(len(np_city)):
        if np_city[i] in citydic:
            citydic[np_city[i]]+=np_casuality[i]
        else:
            citydic[np_city[i]]=np_casuality[i]
    
    count=0
    city=''
    for i in citydic:
        if citydic[i]>count:
            count=citydic[i]
            city=i
    print(city, count)
    del citydic[city]
    
    count=0
    city=''
    for i in citydic:
        if citydic[i]>count:
            count=citydic[i]
            city=i
    print(city, count)
    del citydic[city]
    
    count=0
    city=''
    for i in citydic:
        if citydic[i]>count:
            count=citydic[i]
            city=i
    print(city, count)
    del citydic[city]
    
    count=0
    city=''
    for i in citydic:
        if citydic[i]>count:
            count=citydic[i]
            city=i
    print(city, count)
    del citydic[city]
    
    count=0
    city=''
    for i in citydic:
        if citydic[i]>count:
            count=citydic[i]
            city=i
    print(city, count)
    del citydic[city]
    


# **Most frequent day of attack in a Terrorism Data-set**
# > I have printed the count of frequent day and number of attack as Integer value.

# In[ ]:


with open('../input/terrorismData.csv', encoding='utf8') as file_obj:
    file_data=csv.DictReader(file_obj, skipinitialspace=True)
    
    day=[]
    for row in file_data:
        day.append(row['Day'])
    np_day=np.array(day, dtype='int')
    day, count=np.unique(np_day, return_counts=True)
    print(day[np.argmax(count)], count[np.argmax(count)])


# **ANALYSIS USING PANDAS**

# In[ ]:


#importing important libraries.
import pandas as pd


# **The Most Dangerous city in Jammu and Kashmir and the terrorist group which is most active in that city**
# > Note:Ignoring the Unknown Terrorist Group. Here Dangerous related with the number of terrorist attacks.
# > I will be printing the count of number of attacks in that city as integer value.

# In[ ]:


td=pd.read_csv('../input/terrorismData.csv', encoding='utf8')
df=td.copy()
df=df[df.State=='Jammu and Kashmir']
df=df[df.City==df.City.describe().top]
count=df.shape[0]
df=df[df.Group!='Unknown']
city=df.City.describe().top
group=df.Group.describe().top
print('CITY    ', 'COUNT  ', 'GROUP')
print(city, count, group)


# **Country with Highest Number of Terror Attack and the year in which the most number of terrorist attack happened in that country**
# > I have printed the name of the country, count of terror attacks as integer value and the year in which most number of terriorist attacks happened in that country.

# In[ ]:


df=td.copy()
df=df[df.Country==df.Country.describe().top]
count=df.shape[0]
country=df.Country.describe().top
y={}
for i in df.Year:
    if i in y.keys():
        y[i]+=1
    else:
        y[i]=1
cnt=0
year=0
for i in y.keys():
    if cnt<y[i]:
        cnt=y[i]
        year=i
print(country, count, year)


# **Most Deadliest attack in a history of HumanKind (According to the given dataset)**
# > I have printed count of Killed people as integer value, the country in which the attack happened, and the terriorist group which was involved in the attack.
# Here Deadliest attack means, in which the most number of people killed.

# In[ ]:


df=td.copy()
df=df[df.Killed==df.Killed.max()]
killed=df.Killed.iloc[0]
country=df.Country.iloc[0]
group=df.Group.iloc[0]
print(int(killed), country, group)


# There was formation of new government in India on 26 May 2014. So current government's span is from 26th May 2014 to current.
# So, the two things I have found out from this period are:
# * Total number of attacks done in this period in India.
# * Terrorist group which was most active in this period in India. Most active means, group which has done maximum number of attacks. 

# In[ ]:


df=td.copy()
a=df[df.Day>=26]
b=a[a.Year==2014]
c=b[b.Country=='India']
ans1=c[c.Month==5]
del a
del b
del c
a=df[df.Year==2014]
b=a[a.Country=='India']
ans2=b[b.Month>5]
del a
del b
a=df[df.Country=='India']
ans3=a[a.Year>2014]
count=ans1.shape[0]+ans2.shape[0]+ans3.shape[0]
print(count, end=' ')
ans1=ans1[ans1.Group!='Unknown']
ans2=ans2[ans2.Group!='Unknown']
ans3=ans3[ans3.Group!='Unknown']
print(ans3.Group.describe().top)


# **Frequency of the Casualty in Red Corridor states and in Jammu and Kashmir. Here Frequency is (Total Casualty/Total Number of a year)**
# > Red Corridor state includes Jharkhand, Odisha, Andhra Pradesh, and Chhattisgarh. Here Casualty=Killed +Wounded. I have separately managed the nan values present in the killed and wounded feature.

# In[ ]:


df_terrorism=td.copy()

year=len(set(df_terrorism['Year']))


df_terrorism=df_terrorism[df_terrorism['Country']=='India']

df_terrorism['Casualty']=df_terrorism['Killed']+df_terrorism['Wounded']

Jammu_state=df_terrorism[df_terrorism['State']=='Jammu and Kashmir']

red_state=df_terrorism[(df_terrorism['State']=='Jharkhand')|(df_terrorism['State']=='Odisha')
                       |(df_terrorism['State']=='Andhra Pradesh')|(df_terrorism['State']=='Chhattisgarh')]

red_casualty=int(np.sum(red_state['Casualty']))

Jammu_casualty=int(np.sum(Jammu_state['Casualty']))

print(red_casualty//year,Jammu_casualty//year)

