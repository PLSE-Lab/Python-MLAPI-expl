#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data =pd.read_csv("../input/heart.csv")


# In[2]:


print(data.info())
print(data.describe())


# In[3]:


data.shape


# In[4]:


data.corr()


# In[5]:


data.columns


# In[6]:



a,b = plt.subplots(figsize=(10,10))# figsize meaning is determine size of shape
sns.heatmap(data.corr(),annot=True,linewidths=0.9,fmt='.3f',ax=b) 
plt.show()


# In[7]:


data.head(100).loc[20:25,["slope","oldpeak","target","cp"]]


# In[8]:


new = pd.concat([data.tail(5).loc[::-1,["slope","oldpeak","target","cp"]],data.head(5).loc[::-1,["slope","oldpeak","target","cp"]]],axis = 0)
print(new)


# In[9]:


new["loop"]=["tr_" if ((oldpeak >0.8) & (slope == 1)) else "fal_" for slope,oldpeak,target,cp in new.values]
   


# In[10]:


print(new)


# In[11]:


new.drop(["loop"],axis=1,inplace = True)


# In[12]:


print(new)


# In[13]:


data.slope.plot(kind='line',color='r',label='slope',grid=True,alpha=1,figsize=(5,5),linewidth=0.8,linestyle=':')
plt.legend(loc='lower left')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.title('title')
plt.show()


# In[14]:


data.plot(kind='scatter',color='r',label='scatter',grid=True,alpha=0.9,figsize=(5,5),x='target',y='cp')
plt.legend(loc='upper center')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.title('title')
plt.show()


# In[15]:


data.plot(kind='line',color='r',label='scatter',grid=True,alpha=0.9,x='target',y='cp',linewidth=0.9,linestyle=':')
plt.legend(loc='upper center')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.title('title')
plt.show()


# In[16]:


data.target.plot(kind='hist',color='r',label='hist',grid=True,alpha=1,figsize=(5,5),bins=10)


# In[17]:


#Dictionary

dictionary = {'spain' : 'madrid','usa' : 'vegas'}
print(dictionary.keys())
print(dictionary.values())


# In[18]:


dictionary['spain']='london'
print(dictionary)


# In[19]:


dictionary['france']="paris"
print(dictionary)


# In[20]:


del dictionary['spain']
print(dictionary)


# In[21]:


print('france' in dictionary) 
print(dictionary)


# In[22]:


dictionary.clear()                   # remove all entries in dict
print(dictionary)


# In[23]:


variable_1=data['target']
print(type(variable_1))
variable_2=data[['target']]
print(type(variable_2))


# In[24]:


print(variable_2)


# In[25]:


x=variable_2['target']>0
print(x)


# In[26]:


print(variable_2[x])


# In[27]:


data[(data['target']>0) & (data['cp']>2)]


# In[28]:


print(type(data))


# In[29]:


i=0
while i!=5:
    print('i is = ',i)
    i+=1
print('i is equal', i)
    


# In[30]:


lis = [1,2,3,4,5]
for y in lis:
    print('i is: ',i)
print('')

dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

for index,value in data.loc[0:3,['cp']].iterrows(): 
    print('index = ',index,' value = ',value)


# In[45]:





# In[46]:





# In[31]:





# In[32]:





# In[33]:





# In[34]:





# In[35]:





# In[36]:





# In[37]:





# In[38]:





# In[39]:





# In[40]:





# In[41]:


data.dtypes


# In[42]:





# In[43]:





# In[ ]:





# In[44]:





# In[ ]:




