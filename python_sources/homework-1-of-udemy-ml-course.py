#!/usr/bin/env python
# coding: utf-8

# **First Homework**

# In[ ]:


import numpy as np # Linear Algebra
import matplotlib.pyplot as plt # For plotting graphs
import pandas as pd # For csv file...
import seaborn as sns # Visualization Tool


# In[ ]:


data = pd.read_csv('../input/championsdata.csv')
data.info()


# In[ ]:


data.corr()


# We can see poisitive and negative correlation.
# * For instance  the number of assists made increases as the number increases.

# In[ ]:


# Correlation map
f,ax=plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(),annot=True,linewidths=0.5,fmt='.1f',ax=ax)
plt.show()


# In[ ]:


data.head(10) # Print top 10 data 


# In[ ]:


data.columns # Print features


# In[ ]:


# Line Plot
#color=color, label=label, linewidth= width of line, alpha = opacity, grid=grid, linestyle=linestyle
data.AST.plot(kind='line',color='r',label='Assists',linewidth=1,alpha=1,grid=True,linestyle='-')
data.TOV.plot(color='b',label='Turnover',linewidth=1,alpha=0.5,grid=True,linestyle='-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


# Scatter Plot
# x= Points, y= Win
data.plot(kind='scatter',x='PTS', y='Win',color='red')
plt.xlabel('Points')
plt.ylabel('Win')
plt.title('Points-Win Scatter Plot')


# In[ ]:


# Histogram
# bins=number of bar in figure

data.PTS.plot(kind='hist',bins=50,figsize=(12,12))
plt.show()


# In[ ]:


# clf clean screen
data.PTS.plot(kind='hist',bins=50,figsize=(12,12))
plt.clf() 


# In[ ]:


# Dictionary
dictionary={'Turkey':'Ankara','England':'London'}
print(dictionary.keys())
print(dictionary.values())


# In[ ]:


# Update-Insert*Delete
dictionary['Turkey']='Istanbul'
print(dictionary)
dictionary['france'] = "paris"       # Add new entry
print(dictionary)
del dictionary['Turkey']              # remove entry with key 'spain'
print(dictionary)
print('france' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)


# In[ ]:


# In order to run all code you need to take comment this line
# del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted


# In[ ]:


# PANDAS
series = data['PTS']        # data['Defense'] = series
print(type(series))
print(series)
data_frame = data[['PTS']]  # data[['Defense']] = data frame
print(type(data_frame))
print(series)


# In[ ]:


# Comparison operator
print(3 > 2)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)


# In[ ]:


# 1 - Filtering Pandas data frame
x = data['PTS']>130     # There are only 3 pokemons who have higher defense value than 200
data[x]


# In[ ]:


# 2 - Filtering pandas with logical_and
# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100
data[np.logical_and(data['PTS']>120, data['AST']>30 )]


# In[ ]:


# Same code 
# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['PTS']>120) & (data['AST']>30)]


# In[ ]:


# LOOPS
# Stay in loop if condition( i is not equal 5) is true
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1 
print(i,' is equal to 5')


# In[ ]:


# Stay in loop if condition( i is not equal 5) is true
lis = [1,2,3,4,5]
for i in lis:
    print('i is: ',i)
print('')

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(lis):
    print(index," : ",value)
print('')   

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

# For pandas we can achieve index and value
for index,value in data[['PTS']][0:1].iterrows():
    print(index," : ",value)


# *End of the first homework, thanks for sharing :)*
