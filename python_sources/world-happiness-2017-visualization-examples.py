#!/usr/bin/env python
# coding: utf-8

# # **Visualization of World Happiness 2017**
# 
# In this kernel, I will show you some visualization examples using matplotlib and seaborn libraries.

# In[ ]:


#import the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  

#read the file
data = pd.read_csv('../input/2017.csv') 


# In[ ]:


#print information of data
data.info() #print information of data


# In[ ]:


#shows column names of the data
print(data.columns) 


# Existing column names of the dataset includes . character, so first change column names

# In[ ]:


data.rename(columns={
    'Happiness.Rank': 'Happiness_Rank', 
    'Happiness.Score': 'Happiness_Score',
    'Whisker.high' : 'Whisker_High',
    'Whisker.low' : 'Whisker_Low',
    'Economy..GDP.per.Capita.' : 'Economy_GDP_Per_Capital',
    'Health..Life.Expectancy.' : 'Health_Life_Expectancy',
    'Trust..Government.Corruption.' : 'Trust_Government_Corruption',
    'Dystopia.Residual' : 'Dystopia_Residual'
}, inplace=True)


# In[ ]:


#see the changed column names
print(data.columns) 


# First, we can control the correlations between all columns.

# In[ ]:


#shows correlation matrix of the data
print(data.corr()) 


# To see the correlation in more visual way, we can use heatmap from seaborn library.
# 
# The map shows that there is a positive correlation between Happiness Score&Whisker High&Whisker Low because the values are 1.0
# * The closer the values is to 1.0, the positive correlation is higher.
# * The closer the values is to -1.0, the negative correlation is higher
# 

# In[ ]:


#Show the correlation more visual, with heatmap 
f,ax = plt.subplots(figsize=(10, 10)) #size of the heatmap
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax) 
plt.show()


# We can use different visualisations such as Line Plot, Scatter Plot, Histogram.

# In[ ]:


#Line Plot-Example1
plt.plot(data.Happiness_Score, data.Health_Life_Expectancy, color='green', marker='o', linestyle='-.',
        linewidth=2, markersize=5)
plt.xlabel('Happiness Score')
plt.ylabel('Health Life Expectancy')
plt.title('Line Plot-Relationship of Happiness and Health Life Expectancy')
plt.show()


# In[ ]:


# Line Plot-Example2
data.Family.plot(kind = 'line', color = 'g',label = 'Family',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Health_Life_Expectancy.plot(color = 'r',label = 'Health_Life_Expectancy',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')
plt.title('Line Plot-Family and Health Life Expectancy')
plt.show()


# In[ ]:


# Scatter Plot-Example1
plt.scatter(data.Health_Life_Expectancy,data.Happiness_Score,alpha = 0.5,color = 'green')
plt.xlabel('Health_Life_Expectancy')
plt.ylabel('Happiness Score')
plt.title('Scatter Plot-Happiness Score versus Health_Life_Expectancy')
plt.show()


# It is obviously seen in graphs, there is a positive correlation between Happiness Score and Health Life Expectancy. 
# You can see that when Health Life Expectancy increases, Happiness Score increases too.

# In[ ]:


# Scatter Plot-Example2
data.plot(kind='scatter', x='Trust_Government_Corruption', y='Economy_GDP_Per_Capital',alpha = 0.5,color = 'blue')
plt.xlabel('Trust_Government_Corruption')
plt.ylabel('Economy_GDP_Per_Capital')
plt.title('Scatter Plot-Economy GDP Per Capital versus Trust_Government_Corruption')
plt.show()


# In[ ]:


# Histogram
data.Trust_Government_Corruption.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()

