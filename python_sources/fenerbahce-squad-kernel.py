#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/fenerbahce-squad-database/fenerbahce_footballers.csv')


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data.corr()


# In[ ]:


#correlation map

f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(),annot = True , linewidth = .5, fmt = '.2f',ax = ax)
plt.show()


# In[ ]:


data.head(7)


# In[ ]:


data.tail(5)


# In[ ]:


data.columns


# In[ ]:


#line plot
data.age.plot(kind = "line" , color = "red" , alpha = 0.5 , grid = True , linestyle = ":",label = "age")
plt.legend(loc = "upper right")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Footballer's Ages Line Plot")


# In[ ]:


#scatter plot
data.plot(kind = "scatter" , x = "age" , y = "contract_to" , color = "red")
plt.xlabel("Age")
plt.ylabel("Contract Expired")
plt.title("Age-Contract Scatter Plot")
plt.show()


# In[ ]:


#histogram 
data.age.plot(kind = "hist",bins = 20,figsize = (15,15) )
plt.show()


# In[ ]:


#clf() = cleans it up again you can start fresh
data.age.plot(kind = "hist",bins = 20)
plt.clf()


# In[ ]:


#create dictionary and look its keys and values
footballers = {"Zanka" : "CB" , "Jailson" : "DMF"}
print(footballers.keys())
print(footballers.values())


# In[ ]:


footballers["Jailson"] = "CB" #update existing value
footballers["Altay Bayindir"] = "GK" #insert new record
print(footballers)
print("Zanka" in footballers) #check exist or not
footballers.clear() #clear all dict
print(footballers)


# In[ ]:


#filtering data
filter = data['age'] > 28
data[filter]


# In[ ]:


#filtering with logical and
data[np.logical_and(data['age']<20,data['contract_to']>2020)]


# In[ ]:


for index,value in data[['player']][0:12].iterrows():
    print(index ," : ",value)


# In[ ]:


#list compherension

list_compherension = [i*2 for i in data['age'][0:5]]
list_compherension


# In[ ]:


#list of frequency of players' positions
print(data['position'].value_counts(dropna=False))


# In[ ]:


data1 = data.copy()

average_age = sum(data1.age)/len(data1.age)

data1['age_level'] = ["low" if value < average_age else "high" for value in data1.age ]

data1.age_level


# In[ ]:


#box Plot
data1.boxplot(column = "age" , by = "age_level")
plt.show()


# In[ ]:


#melting
data2 = data.head()
melted = pd.melt(frame = data2,id_vars = 'player',value_vars = ["position" , "nationality"])
melted


# In[ ]:


#pivoting data
#reverse of melting
melted.pivot(index = "player" , columns = "variable" , values = "value")


# In[ ]:


#concatenating data
data_head = data.head()
data_tail = data.tail()
conc_data_row = pd.concat([data_head,data_tail],axis=0,ignore_index=True)
conc_data_row


# In[ ]:


data.dtypes


# In[ ]:


#assert statment
assert 1 == 1 #returns nothing because it is True


# In[ ]:


assert data["position"].notnull().all() #returns nothing because we don't have NaN  values.


# In[ ]:


#HIERARCHICAL INDEXING

data_new = data.copy()


data_new= data_new.set_index(["age",'nationality'])
data_new

