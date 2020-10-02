#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# PS: Before I start my first report, I would like to thank to DATAI Team for their support on my education

# In[ ]:


data = pd.read_csv("../input/2017.csv")


# In this report, I will focus on what kind of situations are effective on happiness level of a country and I am going to use a dataset to understand these criteria. First of all, I would like to start with looking correlations inside data sets.

# In[ ]:


data.corr()


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(13,13))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# As we can observe on the heat map, happiness level (rank or score) is related to a lot of factor of a country.  To understand the dataset better, I prefer to make a quick look to my data below.

# In[ ]:


data.info()


# According to information of my data, we can see that there are 155 countries which are ranked by some criteria like economical level, their freedom etc.

# In[ ]:


data.columns=data.columns.str.replace(".","_")
data.columns=data.columns.str.replace("Economy__GDP_per_Capita_","Economy")
data.columns
data.head()


# With the help of my first 5 data, we can see the top 5 happiest countries in world. Now lets make some plots to look for which criteria is more effective on happiness. First, I would like to consider economical possibilities and trust on goverment of countries according to their happiness ranking.

# In[ ]:


#Lineplot
plt.plot(data.Happiness_Rank,data.Economy, label="GDP per Capita", alpha=0.8, linewidth=1)
plt.plot(data.Happiness_Rank,data.Trust__Government_Corruption_,color="red",label="Goverment Trust",alpha=0.8,linewidth=1)
plt.legend()
plt.xlabel("Happiness Rank")
plt.show()


# As we can see cleary on figure above, economical situation of citizens has critical effect on happiness level. When you consider challenges of huminity for survival, you can say money is one of the most important thing to solve problems in our lives. So, it is not hard to understand that happiness ranking is directly propotional to economical level. However, if we look the red line which presents govermental trust between citizen, we can say whether it is a happy country or not, most of the governments are not able to get their own citizen trusts. 
# Lets have a look to how freedom feeling affects the hapiness score.

# In[ ]:


# Scatter plot

data.plot(kind='scatter', x='Freedom', y='Happiness_Score',figsize=(10,10), alpha = 0.5,color = 'red', s=200)

plt.xlabel('Freedom')             
plt.ylabel('Happiness Score')
plt.title("Freedom's effect on Happiness Score")   
plt.show()


# Not so general but we cannot deny that freedom feeling is increasing with happiness score. Now, lets look for average happiness score of these countries that can help us to seperate our data a bit.

# In[ ]:


print(data.Happiness_Score.describe())

happy_country=data[data["Happiness_Score"]>5.354]
unhappy_country=data[data["Happiness_Score"]<5.354]


# When we have the mean value for happiness score, I would like to seperate these countries as happy and unhappy ones. This seperation can give us better point of view when we consider differences between countries. 
# In the figure below, I compare the freedom level for these two categories and as we can see, happy countries obviously have better freedom feeling than unhappy ones. So, it is clear to say that freedom is one of the feeling that makes us feel better.

# In[ ]:


#Histogram
happy_country.Freedom.plot(kind="hist",bins=50, figsize=(10,10),color="yellow")
unhappy_country.Freedom.plot(kind="hist",bins=50,figsize=(10,10))
plt.show()


# In[ ]:


for index,value in enumerate(happy_country.Economy):
    if value >1.5:
       print(happy_country.Happiness_Rank[index],happy_country.Country[index])
economical_proof=data[data["Economy"]>1.5]
economical_proof.head(100)


# According to my data, I assumed that a country can be considered as rich if their economy per capita is higher than 1.5 and I can see that all these rich countries are on top part of our mean line. So, as I showed before by using line plot, I can show economy effect on happiness score again on scatter plot

# In[ ]:



data.plot(kind="scatter", x="Economy", y="Happiness_Score",figsize=(10,10),color="purple", s = 200)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




