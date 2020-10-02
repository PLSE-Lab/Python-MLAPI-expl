#!/usr/bin/env python
# coding: utf-8

# ## World Happiness Report 2016 Data Exploration

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


happiness_report_2015 = pd.read_csv('../input/2015.csv')
happiness_report_2016 = pd.read_csv('../input/2016.csv')
happiness_report_2015.head()


# ## Plotting Pearson's Correlation
# 
# - I would like to know which of the variables with high correlation coefficient (+ve / -ve) especially to the Happiness Score, with hopes of finding something interesting.

# In[ ]:


# Plotting heatmap of pearson's correlation for 2015
fig, axes = plt.subplots(figsize=(10, 7))
corr = happiness_report_2015.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr,linewidths=1,annot=True, mask=mask, vmax=.3, square=True)
axes.set_title("2015")


# ### From the matrix plot above the variables with the highest correlation coefficient in 2015 are:
# 
# - Economy (GDP per Capita) -> Happiness Score | 0.78
# - Family -> Happiness Score | 0.74
# - Family -> Economy (GDP per Capita) | 0.65
# - Health (Life Expentacy) -> Happiness Score |0.72

# In[ ]:


sns.pairplot(happiness_report_2015[['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)']])


# In[ ]:


# Plotting heatmap of pearson's correlation for 2016
fig, axes = plt.subplots(figsize=(10, 7))
corr = happiness_report_2016.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr,linewidths=1,annot=True, mask=mask, vmax=.3, square=True)
axes.set_title("2016")


# ### From the matrix plot above the variables with the highest correlation coefficient in 2016 are:
# 
# - Economy (GDP per Capita) -> Happiness Score | 0.79
# - Family -> Happiness Score | 0.74
# - Family -> Economy (GDP per Capita) | 0.67
# - Health (Life Expentacy) -> Happiness Score |0.76

# In[ ]:


sns.pairplot(happiness_report_2016[['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)']])


# In[ ]:


#plt.plot(happiness_report_2015['Happiness Score'])
fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(10, 7))
sns.distplot(happiness_report_2015['Happiness Score'],kde=True,ax=axes[0])
sns.distplot(happiness_report_2016['Happiness Score'],kde=True,ax=axes[1])
axes[0].set_title("Distribution of Happiness Score for 2015")
axes[1].set_title("Distribution of Happiness Score for 2016")


# ## Comparing Happiness Scores in different Regions between 2015 & 2016

# In[ ]:


happiness_report_2015['Year'] = '2015'
happiness_report_2016['Year'] = '2016'
happiness_report_2015_2016 = pd.concat([happiness_report_2015[['Happiness Score','Region','Year']],happiness_report_2016[['Happiness Score','Region','Year']]])
happiness_report_2015_2016.head()


# In[ ]:


sns.set(font_scale=1.5)
fig, axes = plt.subplots(figsize=(20, 9))
sns.boxplot(y='Region',x='Happiness Score',hue='Year', data = happiness_report_2015_2016)


# From the boxplot above, only 2 regions witnessed a significant increase in happiness score namely;
# - Central and Eastern Europe
# - Southern Asia
# 
# *while* the remaining 3 regions that witnessed a significant decrease in happiness score are;
# - Southeastern Asia
# - Eastern Asia
# - Sub-Saharan Africa
# 

# Comments and Suggestions are appreciated :)

# In[ ]:




