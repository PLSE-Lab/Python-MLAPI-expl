#!/usr/bin/env python
# coding: utf-8

# <font size=6.5>World Happiness Survey</font>

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df = pd.read_csv("../input/2015.csv")
df.head(5)


# <font size=5><b>Regionwise happiness score</b></font>

# In[63]:


plt.bar(df['Region'],df['Happiness Score'],color='green')
plt.xticks(rotation=90)


# <font size=5><b>Effect of Happiness Score on Income</b></font>

# In[62]:


plt.scatter(df['Happiness Score'],df['Economy (GDP per Capita)'],color='green')
plt.xlabel("Happiness Score")
plt.ylabel("Economy")


# <font size=5><b>Happiness Score vs Generosity</b></font>

# In[64]:


plt.scatter(df['Happiness Score'],df['Generosity'],color='red') 
plt.xlabel("Happiness Score")
plt.ylabel("Generosity")


# <font size=5><b>Heatmap</b></font>

# In[39]:


corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


# <b><font size=5>Strip Plot(Region vs Happiness rank)</font></b>

# In[38]:


g = sns.stripplot(x="Region", y="Happiness Rank", data=df, jitter=True)
plt.xticks(rotation=90)


# <b><font size=5>Pair Plot</font></b>

# In[48]:


selectCols=  ['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Region']
sns.pairplot(df[selectCols], hue='Region',size=2.5)


# <font size=5>Comparison of Different Asian Regions</font>

# In[55]:


SoutheasternAsia=df[df.Region=='Southeastern Asia']
SouthernAsia=df[df.Region=='Southern Asia']
EasternAsia=df[df.Region=='Eastern Asia']
f, axes = plt.subplots(3, 2, figsize=(16, 16))
axes = axes.flatten()
compareCols = ['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)']
for i in range(len(compareCols)):
    col = compareCols[i]
    axi = axes[i]
    sns.distplot(SoutheasternAsia[col],color='blue' , label='SouthEastern', ax=axi,rug='True')
    sns.distplot(SouthernAsia[col],color='green', label='Southern',ax=axi,rug='True')
    sns.distplot(EasternAsia[col],color='red',label='Eastern',ax=axi,rug='True')
    axi.legend()

