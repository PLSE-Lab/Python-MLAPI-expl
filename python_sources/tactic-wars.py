#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataFrame = pd.read_csv("../input/tft_database.csv")


# In[ ]:


dataFrame.info()


# In[ ]:


dataFrame.describe()


# In[ ]:


dataFrame.shape


# In[ ]:


dataFrame.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(dataFrame.corr(),annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


dataFrame.head()


# In[ ]:


dataFrame.tail()


# In[ ]:


dataFrame.columns


# In[ ]:


dataFrame.Damage.plot(kind="line",color="red",label="HP",linewidth=1.5,alpha=0.5,grid=True,linestyle="-")
dataFrame.Mana.plot(color="blue",label="Mana",linewidth=1,alpha=1,grid=True,linestyle="-.")
plt.legend()
plt.xlabel("x label")
plt.ylabel("y label")
plt.title("Damage-Mana Line Plot")
plt.show()


# In[ ]:


dataFrame.plot(kind="scatter",x="AttackSpeed",y="DamagePerSecond",alpha=0.5,color="red")
plt.xlabel("Attack Speed")
plt.ylabel("Damage Per Second")
plt.title("AttackSpeed and DamagePerSecond Scatter Plot")
plt.show()


# In[ ]:


dataFrame.HP.plot(kind="hist",bins=20,figsize = (15,15))
plt.show()


# In[ ]:


# clf() = cleans it up again you can start a fresh
dataFrame.AttackSpeed.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()


# In[ ]:


#create dictionary and look its keys and values
dictionary = {"Yasuo":"400","Rengar":"350","ChoGat":"1000"}

print(dictionary.keys())
print(dictionary.values())


# In[ ]:


# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique

dictionary['Yasuo'] = "450" #update existing entry

dictionary['Ahri'] =  "300"  #add new entry

#del dictionary['ChoGath']

print('ChoGath' in dictionary)

dictionary.clear()
print(dictionary)


# In[ ]:


#Filtering data
filtered_data = dataFrame['Damage'] > 70
dataFrame[filtered_data]


# In[ ]:


dataFrame[np.logical_and(dataFrame['HP'] > 700, dataFrame['Mana']>50)]


# In[ ]:


# We can use for loop to achive key and value of dictionary.

dictionary = {"Yasuo":"400","Rengar":"350","ChoGat":"1000"}

for key,value in dictionary.items():
    print(key,":",value)
    
          
# For pandas we can achieve index and value

for index,value in dataFrame[['DamagePerSecond']][0:1].iterrows():
    print(index," : ",value)


# In[ ]:


#List Compherension

list_compherension = [i*2 for i in dataFrame[['HP']][0:-1].iterrows()]
list_compherension


# In[ ]:


data = dataFrame.copy()

average_damage = sum(data.Damage)/len(data.Damage)

data['DamageLevel'] = ["low" if value < average_damage else "high" for value in data.Damage ]

data.DamageLevel


# In[ ]:


#EXPLORATORY DATA ANALYSIS
#value_counts() method

numberOfTypes = dataFrame['Origin'].value_counts(dropna="False")
numberOfTypes


# In[ ]:


#VISUAL EXPLORATORY DATA ANALYSIS
#Box Plot


data.boxplot(column="Damage",by="DamageLevel")
plt.show()


# In[ ]:


#TIDY DATA

data_head = data.head()
data_head


# In[ ]:


#Melting

melted = pd.melt(frame=data_head,id_vars="Champion",value_vars=['Origin','Class'])
melted


# In[ ]:


#Pivoting Data
#Reversing melt

reverse_melt = melted.pivot(index="Champion",columns='variable',values='value')
reverse_melt


# In[ ]:


#CONCATENATING DATA

data_head = data.head()
data_tail = data.tail()

concatenating_data = pd.concat([data_head,data_tail],axis=0,ignore_index="True")
concatenating_data


# In[ ]:


data.dtypes


# In[ ]:


#converting data types

data['Origin'] = data['Origin'].astype('category')
data['Armor'] = data['Armor'].astype('float')


# In[ ]:


data.dtypes #checking whether the type changed or not.


# In[ ]:


#MISSING DATA and TESTING WITH ASSERT

data.info()


# In[ ]:


data['Class2'].value_counts(dropna=False) #We see that there are 47 NaN values.


# In[ ]:


data['Class2'].dropna(inplace=True)


# In[ ]:


#So does it work?
#Let's check it with assert statement

assert 1==1 #returns nothing because it's true


# In[ ]:


assert data['Class2'].notnull().all() #returns nothing because we do not have NaN values.


# In[ ]:


#example data frames from dictionaries

country = ['Turkey','Germany']
capitalCity = ['Ankara','Munich']
list_label=['country','capitalCity']
list_column=[country,capitalCity]
zipped = list(zip(list_label,list_column))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df


# In[ ]:


# Add new columns

df['population'] = [80,65]
df


# In[ ]:


#Broadcasting

df['income'] = 1
df


# In[ ]:


#VISUAL EXPLORATORY DATA ANALYSIS

#plotting all data

data1=data.loc[:,["Damage","DamagePerSecond","Mana"]]
data1.plot()
plt.show()

#It's a little confusing


# In[ ]:


data1.plot(subplots=True)
plt.show()


# In[ ]:


#INDEXING PANDAS TIME SERIES



date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data_head['date'] = datetime_object

# lets make date as index

data_head = data_head.set_index("date")

data_head



# In[ ]:


print(data_head.loc["1992-01-10 "])


# In[ ]:


data_head.resample("A").mean()


# In[ ]:


data_head.resample("M").mean()


# In[ ]:


data_head.resample("M").mean().interpolate("linear")


# In[ ]:


# Plain python functions

def multiply(n):
    return n*2

data.Cost.apply(multiply)

data['Cost']
    


# In[ ]:


# Or we can use lambda function

data.Cost.apply(lambda n:n*2)


# In[ ]:


# Defining column using other columns

data['TotalDefense'] = data.Armor + data.MagicResist
data['TotalDefense']


# In[ ]:


#HIERARCHICAL INDEXING

data2 = dataFrame.copy()


data2= data2.set_index(["Origin",'Class'])
data2


# In[ ]:




