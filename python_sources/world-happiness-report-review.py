#!/usr/bin/env python
# coding: utf-8

# **World Happiness Report Review**
# 
# **Abstract**
# 
# In this work we will examine UN's World Hapiness Report. In this report I want to more closely at some stage Turkey.
# Let's define libraries and look dataset.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns #Visulation Tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Data for 3 years : 2015,2016,2017 . Let's read them.

# In[ ]:


data15 = pd.read_csv('../input/2015.csv') # Data Read From CSV
data16 = pd.read_csv('../input/2016.csv')
data17 = pd.read_csv('../input/2017.csv')


# In[ ]:


data15.info() #Data info


# In[ ]:


data15.corr()  # Let's look at the relationship between columns.


# **Correlation Map**
# 
# We looked relationship columns with **data.15corr()** . We will show this with colours. For correct operation you should import  "**matplotlib**" ve "**seaborn**" libraries. 

# In[ ]:


f,ax = plt.subplots(figsize=(25,25))  #Map size.
sns.heatmap(data15.corr(), annot=True,linewidths=.5,fmt='.2f',ax=ax) 
# anot=Numbers Appearence,fmt=Digits After a Comma
plt.show()


# In[ ]:


data15.head(10)  # Top 10 line display 


# In[ ]:


data15.columns #Columns display


# Variables with spaces cause a error. For example "**Standard Error**" , we should change with "**Standard_Error**" so we won't get errors.

# In[ ]:


data15 = data15.rename(index=str, columns={"Happiness Rank": "Happiness_Rank", "Happiness Score": "Happiness_Score", "Standard Error": "Standard_Error", "Economy (GDP per Capita)": "Economy_GDP_per_Capita", "Dystopia Residual": "Dystopia_Residual","Trust (Government Corruption)": "Trust_Government_Corruption",'Health (Life Expectancy)': "Health_Life_Expectancy",})
data15.columns


# What is the relationship between countries happiness and their economies ?
# Does money bring happiness ?
# If bring , How many ?
# Let's look together.
# You can see the effect of all variables on the happiness score.
# You can also see the changes between the variables.

# In[ ]:


# Line Plot 

data15.Happiness_Score.plot(kind='line', color='r',label='Happiness Score', linewidth=2,alpha=1,grid=True,linestyle='-',figsize=(20, 20))
data15.Economy_GDP_per_Capita.plot(kind='line', color='b',label='Economy GDP Per Capita', linewidth=2,alpha=1,grid=True,linestyle='-')
data15.Family.plot(kind='line', color='g',label='Family', linewidth=2,alpha=1,grid=True,linestyle='-')
data15.Health_Life_Expectancy.plot(kind='line', color='black',label='Health_Life_Expectancy', linewidth=2,alpha=1,grid=True,linestyle='-')
data15.Freedom.plot(kind='line', color='orange',label='Freedom', linewidth=2,alpha=1,grid=True,linestyle='-')
data15.Trust_Government_Corruption.plot(kind='line', color='brown',label='Trust_Government_Corruption', linewidth=2,alpha=1,grid=True,linestyle='-')
data15.Generosity.plot(kind='line', color='gray',label='Generosity', linewidth=2,alpha=1,grid=True,linestyle='-')
data15.Dystopia_Residual.plot(kind='line', color='pink',label='Dystopia_Residual', linewidth=2,alpha=1,grid=True,linestyle='-')


plt.legend(loc = 'upper right')
plt.xlabel('Country')
plt.ylabel('Effect Value')
plt.title('World Happiness Score and Affecting Factors')
plt.show()


# Let's draw a chart for the Comparison of Family and Economic Happiness and comment it.

# In[ ]:


#Scatter Plot
# x = Economy_GDP_per_Capita y = Family
data15.plot(kind='scatter', x='Economy_GDP_per_Capita', y='Family',alpha=0.5,color='r')
plt.xlabel('Economy_GDP_per_Capita')
plt.ylabel('Family')
plt.title('Comparison of Family and Economic Happiness')
plt.show()


# Now let's comment on this chart.
# For example; happiness from the family 0 (zero) in case of looking at the value of happiness from the economy when we look at "**A happy family, the economy is bad, the economy is not happy in a family is not happy**" we can easily conclude.
# In the same way, we see that happy families are happy in general and also economically .

# In[ ]:


#Histogram , bins = number of bar figure
data15.Economy_GDP_per_Capita.plot(kind='hist',bins=40,figsize=(12,12))
plt.show()


# What do you see on this chart?
# When you look at those who are very happy with the economy (ie between 1.50 and 1.75 range) you can see that the frequency is less than the others.
# We see that the number of those who are very happy about the economy in the world is very low.

# According to data from Turkey's year Let's get the table and we'll show them.

# In[ ]:


data15_Turkey = data15['Country']== 'Turkey' 
data16_Turkey = data16['Country']== 'Turkey'
data17_Turkey = data17['Country']== 'Turkey'

data15[data15_Turkey].head()


# In[ ]:


data16[data16_Turkey].head()


# In[ ]:


data17[data17_Turkey].head()


# I will continue to improve my work. Waiting for your comments.
