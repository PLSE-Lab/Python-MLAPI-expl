#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#hello every one!! firstly i want to thank you Kaan Can. 
#i will try to analysis pokemon data here. 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

from numpy.random import rand

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


my_data = pd.read_csv('../input/pokemon.csv')


# In[ ]:


my_data.head(7)


# In[ ]:


cropped_data=my_data.drop(columns='#')
cropped_data.head(7)


# In[ ]:


data=cropped_data
data.columns=[i.replace('.','') for i in data.columns]
data.columns=[i.replace(' ','_') for i in data.columns]
data.columns=[i.lower() for i in data.columns]
data.columns


# In[ ]:


data.info()


# In[ ]:


c_data=data.loc[:,'hp':'speed'] #i cant add in loc function columns index for ex. c_data=data.loc[:,3:8] i get error :(
c_data.corr()


# In[ ]:


#heatmap visualization map
plt.subplots(figsize=(11,10))

sns.heatmap(c_data.corr(),vmin=0,vmax=1,annot=True,cbar=True,cbar_kws={"orientation": "horizontal"})
#plt.tight_layout()
plt.show()


# In[ ]:


ad_data=data.loc[:,["name","attack","defense"]] 
#print(ad_data)
names=ad_data.name
attack=ad_data.attack
defense=ad_data.defense
print(ad_data.min())
print(ad_data.max())

af1=(ad_data.attack>=0) & (ad_data.attack<=55)
df1=(ad_data.defense>=0) & (ad_data.defense<=55)
af2=(ad_data.attack>=56) & (ad_data.attack<=90)
df2=(ad_data.defense>=56) & (ad_data.defense<=90)
af3=(ad_data.attack>=91) 
df3=(ad_data.defense>=91) 

fd1=ad_data[af1 & df1]
fd2=ad_data[af2 & df2]
fd3=ad_data[af3 & df3]



#print(attack1)

#data.loc[:20,["name","attack","defense"]]


# In[ ]:


f,ax=plt.subplots(figsize=(25,8))


ax.plot(fd1.name,fd1.attack,'^',color='r',label='Attack Power')
ax.plot(fd1.name,fd1.defense,'v',color='g',label='Defense Power')
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",rotation_mode="anchor",size='10')
plt.text(70,30,'Attack and Defense Power between 0-55',horizontalalignment='center',verticalalignment='center',alpha=0.3,size='50')
plt.xlabel("Pokemons",size='12')
plt.ylabel("Attack and Defense",size='12')
plt.legend()
plt.title("Attack Defense Table")


# In[ ]:


f,ax=plt.subplots(figsize=(30,8))
ax.plot(fd2.name,fd2.attack,'*',color='r',label='Attack Power')
ax.plot(fd2.name,fd2.defense,'o',color='g',label='Defense Power')
ax.grid(True)
plt.text(30,75,'Attack and Defense Power between 55-90',alpha=0.3,size='50')
plt.xticks(fd2.name,rotation='vertical')
#plt.xticks(np.arange(0,len(fd2.name),step=1),fd2.name,rotation='vertical')
#plt.setp(ax.get_xticklabels(), rotation=90, ha="right",rotation_mode="anchor",size='10')
# plt.grid(True)
plt.xlabel("Pokemons",size='12')
plt.ylabel("Attack and Defense",size='12')
plt.legend()
plt.title("Attack Defense Table")


# In[ ]:


f,ax=plt.subplots(figsize=(30,8))

n = 750
scale = 200.0 * rand(n)
    
ax.scatter(fd3.name, fd3.defense, c='g', s=scale, label='Defense',alpha=0.7, edgecolors='b')
ax.scatter(fd3.name, fd3.attack, c='r', s=scale, label='Attack',alpha=0.7, edgecolors='b')
ax.legend(loc=2)
ax.grid(True)

plt.xticks(fd3.name,rotation='vertical')
plt.xlabel("Pokemons",size='12')
plt.ylabel("Attack and Defense",size='12')
plt.title("Attack Defense Table")





# In[ ]:


f, (ax1,ax2)=plt.subplots(2,1,figsize=(20,15))
ax1.hist(attack,color='r',alpha=0.7,bins=50,     histtype='bar', align='mid',         orientation='vertical',   )
ax1.set_xlabel('Atack Power')
ax1.set_ylabel('Unit')
ax1.set_title('ATTACK POWER TABLE')
ax1.grid(True)
ax2.hist(defense,bins=50)
ax2.set_xlabel('Defense Power')
ax2.set_ylabel('Unit')
ax2.set_title('DEFENSE POWER TABLE')
ax2.grid(True)


# In[ ]:


values={}


for i in range(len(ad_data)):
    diff=ad_data.attack[i]-ad_data.defense[i]
#     print(diff)
    values[ad_data.name[i]]=diff
    
name_list=list(values.keys())
diff_list=list(values.values())


plt.figure(figsize=(40,15))
plt.bar(values.keys(),values.values(),color='r',lw=0.5)
plt.xticks(names,rotation='vertical',size=5)
plt.grid(True)



ax.grid(True)




plt.show()


# In[ ]:


data.columns


# In[ ]:


data[data.sp_atk >100].defense.min()
d1=data[data.sp_atk >100].defense


# In[ ]:


data[data.sp_def >100].attack.min()
a1=data[data.sp_def >100].attack


# In[ ]:


s1=data[data.sp_atk >100].speed


# In[ ]:


cor_sp=pd.concat([d1,a1,s1],axis=1)
cor_sp


# In[ ]:


plt.subplots(figsize=(10,10))
sns.heatmap(cor_sp.corr(),vmin=0,vmax=1,annot=True,cbar=True,cbar_kws={"orientation": "horizontal"})

