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
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/pokemon.csv')


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


# lets return pokemon csv and make one more list comprehension example
# lets classify pokemons whether they have high or low speed. Our threshold is average speed.
threshold = sum(data.Speed)/len(data.Speed)
data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]
data.loc[:10,["speed_level","Speed"]] # we will learn loc more detailed later


# In[ ]:


#Let s look at the frequency of pokemons feature
print (data['Type 1'].value_counts(dropna=False)) #also nan values are counted by dropna=False


# In[ ]:


data.boxplot(column='Attack',by = 'Legendary')
plt.xlabel('Legendary')
plt.ylabel('Attack')
#plt.title('Boxplot of Attack-Legendary')
plt.show()


# In[ ]:


data_new = data.head()    # I only take 5 rows into new data
data_new


# **MELTED**
# 
# Whatever you want to do with columns  which is combine into one column diffirent 2 or 3 columns but watch out id_vars point out the singular(primary key)

# In[ ]:


melted=pd.melt(frame=data_new, id_vars='Name',value_vars=['Attack','Defense'])
melted


# In[ ]:


#reverse melting
melted.pivot(index = 'Name', columns = 'variable',values='value')


# In[ ]:


# CONCATENATING DATAFRAMES
data1 = data.head()
data2= data.tail()
conc_data = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 (horizontal) : adds row by row 
conc_data


# In[ ]:


data1 = data['Speed'].head()
data2= data['Defense'].head()
conc_data_v = pd.concat([data1,data2],axis =1) # axis = 1 (vertical) : adds column by column 
conc_data_v


# object(string) - to - Category --https://www.tutorialspoint.com/python_pandas/python_pandas_categorical_data.htm
# float - to - int

# In[ ]:


data.dtypes #show us colums type


# In[ ]:


#some example
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')


# In[ ]:


data.dtypes


# In[ ]:


data.info()# as you can see 1 null in the name, 386 null value of Type 2 


# In[ ]:


data["Type 2"].value_counts(dropna =False)#count all of type 2 and 386 nan(null) value


# In[ ]:


data1=data   
data1["Type 2"].dropna(inplace = True)#drop nan values


# In[ ]:


data["Type 2"].fillna('empty',inplace = True)


# In[ ]:


assert  data['Type 2'].notnull().all()# if this statement> data1['Type 2'].notnull().all() is wrong than retrun false 
#but if it is true returns nothing because we drop nan values


# **NOTE**
# 
# With assert statement we can check a lot of thing. For example
#  
# assert data.columns[1] == 'Name'
# 
#  assert data.Speed.dtypes == np.int
#  
# 

# **PANDAS FOUNDATION**
# 
#  single column = series
#  
# NaN = not a number
# 
# dataframe.values = numpy

# In[ ]:


# data frames from dictionary
#ste by step we can see  whats happening.
country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
print (type(list_label))

list_col = [country,population]
print (list_col)
print (type(list_col))

zipped = list(zip(list_label,list_col))
print (zipped)
print (type(zipped))

data_dict = dict(zipped)
print(data_dict)
print (type(data_dict))

df = pd.DataFrame(data_dict)
print (type(df))
df


# In[ ]:


# Add new columns
df["capital"] = ["madrid","paris"]
df


# In[ ]:


# Broadcasting
df["income"] = 0 #Broadcasting entire column
df


# In[ ]:


data.head()


# In[ ]:


data1 = data.loc[:,["Attack","Defense","Speed"]]#show us overleapping 3 column grafics
data1.plot()


# In[ ]:


# subplots
data1.plot(subplots = True)#differentiate by the subplots
plt.show()


# In[ ]:


# scatter plot  
data1.plot(kind = "scatter",x="Attack",y = "Defense")
plt.show()


# In[ ]:


# hist plot  
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)#normed(boolean): normalize or not


# In[ ]:


# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)#cumulative(boolean): compute cumulative distribution
plt.savefig('graph.png')
plt


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




