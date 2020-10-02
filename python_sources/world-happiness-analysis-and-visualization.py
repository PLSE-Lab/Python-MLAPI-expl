#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


path1 = "../input/world-happiness/2015.csv"
y_5 = pd.read_csv(path1)


# In[ ]:


import math 
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


y_5.head()


# In[ ]:


y_5.info()


# In[ ]:


y_5.describe()


# # **Exploratory Data Analysis and Visualization**

# **Before we start our analysis, the most confusing thing you would ever encoounter is from where to start. Basically, there is so much data to look upon that it's quite normal to be confused, but, not anymore. Firstly, if you have no idea about the features ,tackle the target variable(in our case, Happiness score or rank), then think about what factors are responsible for this trend in that variable(Check correlation between the variables through Heatmap). Once you would do these things, you would gain slight insights from the data which would be result in more relations and once, this flow is started, you would be unstoppable!**

# In[ ]:


# Happiness score across Regions

plt.figure(figsize=(12,9))
sns.violinplot(y_5['Happiness Score'], y_5['Region'])
plt.show()


# In[ ]:


# Getting Countries with more than 84% of Happiness scores
reg_5 = y_5.loc[y_5['Happiness Score'] > 6.3]
reg_5


# In[ ]:


# Regions with most no. of happy countries
reg_5['Region'].value_counts()


# In[ ]:


# Getting countries in regions having 6.3+ scores
west_euro = y_5.loc[(y_5['Region'] == 'Western Europe') & (y_5['Happiness Score'] > 6.3)] 
print('Total countries in Western Europe with more than 6.3+ score :',len(west_euro.index))

latin_america = y_5.loc[(y_5['Region'] == 'Latin America and Caribbean') & (y_5['Happiness Score'] > 6.3)]
print('Total countries in Latin America and Caribbean with more than 6.3+ score :',len(latin_america.index))

middle_east = y_5.loc[(y_5['Region'] == 'Middle East and Northern Africa') & (y_5['Happiness Score'] > 6.3)]
print('Total countries in Middle East and Northern Africa with more than 6.3+ score :',len(middle_east.index))

southeast_asia = y_5.loc[(y_5['Region'] == 'Southeastern Asia') & (y_5['Happiness Score'] > 6.3)]
print('Total countries in Southeastern Asia with more than 6.3+ score :',len(southeast_asia.index))


# In[ ]:


# Creating a dataframe containing regions with countries having 6.3+ scores
top_4 = pd.DataFrame({'Region':['Western Europe', 'Latin America and Caribbean', 'Middle East and Northern Africa', 'Southeastern Asia']
                      ,'Countries':[len(west_euro.index), len(latin_america.index), len(middle_east.index), len(southeast_asia.index)]})

# Visualizing this dataframe
plt.figure(figsize=(10,8))
sns.barplot(x=top_4['Region'],y=top_4['Countries'],data=top_4)
plt.ylabel('Countries with 6.3+ scores')
plt.xlabel('Regions')
plt.xticks(rotation = 75)
plt.title('Regions containing most no. of countries with 6.3+ scores')


# In[ ]:


#  Getting top 10 happiest countries in the world
happy_10 = y_5.sort_values(by='Happiness Score',ascending=False).head(10)
plt.figure(figsize=(15,10))
sns.barplot(x='Happiness Score',y='Country',data=happy_10,hue='Country')
plt.legend(loc='lower right')
plt.xlabel('Happiness Scores')
plt.ylabel('Countries')
plt.title('Top 10 happiest countries in 2015')
plt.show()


# *Switzerland is the happiest country with a happiness score of 7.57(approx.)*

# In[ ]:


# Getting top 10 unhappy countries in the world
sad_10 = y_5.sort_values(by='Happiness Score',ascending=False).tail(10)
plt.figure(figsize=(15,10))
sns.barplot(x='Happiness Score',y='Country',data=sad_10,hue='Country')
plt.legend(loc='lower right')
plt.xlabel('Happiness Scores')
plt.ylabel('Countries')
plt.title('Top 10 unhappy countries in 2015')
plt.show()


# *Togo is the most unhappy country with a happiness score of 2.8(approx.)*

# **Now, let's take a look at what Economy has to do with Happiness.**

# In[ ]:


#  Getting top 10 happiest countries' Economy in the world
happy_10 = y_5.sort_values(by='Happiness Score',ascending=False).head(10)
plt.figure(figsize=(15,10))
sns.barplot(x='Economy (GDP per Capita)',y='Country',data=happy_10,hue='Country')
plt.legend(loc='lower right')
plt.xlabel('Economy (GDP per Capita)')
plt.ylabel('Countries')
plt.title('Top 10 happiest countries with their Economy in 2015')
plt.show()


# *The most surprising thing I have seen in this plot is that it is not necessary that the most developed nations will also be the most happiest.There are nations with more GDP per capita than Switzerland but still behind him.*

# In[ ]:


# Getting top 10 unhappy countries' Economy in the world
sad_10 = y_5.sort_values(by='Happiness Score',ascending=False).tail(10)
plt.figure(figsize=(15,10))
sns.barplot(x='Economy (GDP per Capita)',y='Country',data=sad_10,hue='Country')
plt.legend(loc='lower right')
plt.xlabel('Economy (GDP per Capita)')
plt.ylabel('Countries')
plt.title('Top 10 unhappy countries with their Economy in 2015')
plt.show()


# *Syria has a decent GDP per capita of around 0.75 but still it is the top 3 unhappy countries(maybe because of the Trust and Freedom, we will explore it later).*

# In[ ]:


#  Getting top 10 happiest countries' Generosity in the world
happy_10 = y_5.sort_values(by='Happiness Score',ascending=False).head(10)
plt.figure(figsize=(15,10))
sns.barplot(x='Generosity',y='Country',data=happy_10,hue='Country')
plt.legend(loc='lower right')
plt.xlabel('Generosity')
plt.ylabel('Countries')
plt.title('Top 10 happiest countries with their Generosity in 2015')
plt.show()


# *Confusing trends have been shown by Generosity, hence, no conclusion can be drawn.*

# In[ ]:


#  Getting top 10 unhappy countries' Generosity in the world
sad_10 = y_5.sort_values(by='Happiness Score',ascending=False).tail(10)
plt.figure(figsize=(15,10))
sns.barplot(x='Generosity',y='Country',data=sad_10,hue='Country')
plt.legend(loc='lower right')
plt.xlabel('Generosity')
plt.ylabel('Countries')
plt.title('Top 10 unhappy countries with their Generosity in 2015')
plt.show()


# *Though Syria is showing some great trends, still it is placed in the worst 3 according to happiness rank. We will discover the reason for the placement of Syria in the worst 3, later in this notebook.*

# **Bivariate Analysis**

# In[ ]:


# Checking correlation between different variables
plt.figure(figsize=(15,12))
sns.heatmap(y_5.corr(), cmap = 'copper', annot = True)
plt.show()


# *The highly correlated features with happiness scores are Economy, Family and Health. Other important factors to keep in mind are Freedom and Dystopia Residual.*

# * **Correlation for Western Europe**

# In[ ]:


plt.figure(figsize=(15,12))
west_europe = y_5.loc[lambda y_5 : y_5['Region'] == 'Western Europe']
sns.heatmap(west_europe.corr(), cmap = 'Greys', annot = True)
plt.show()


# *It seems that the highly correlated features are Family, Freedom and Trust in Western Europe*

# In[ ]:


sns.jointplot('Family', 'Freedom', data=west_europe, kind='kde', space=0)


# In[ ]:


sns.jointplot('Trust (Government Corruption)', 'Economy (GDP per Capita)', data=west_europe, kind='kde', space=0, color='g')


# *This looks like a normal (Gaussian) distribution.*

# In[ ]:


sns.jointplot('Trust (Government Corruption)', 'Freedom', data=west_europe, kind='kde', space=0)


# *Looks like Freedom is highly correlated with Family and Trust(So, if you want to predict happiness scores, you better eliminate Freedom as it could lead to multicollinearity).*

# * **Correlation for Eastern Asia**

# In[ ]:


plt.figure(figsize=(15,12))
east_asia = y_5.loc[lambda y_5 : y_5['Region'] == 'Eastern Asia']
sns.heatmap(east_asia.corr(), cmap = 'pink', annot = True)
plt.show()


# *In Eastern Asia, Economy and Health seems to be the most important factors responsible for Happiness.*

# In[ ]:


sns.jointplot('Health (Life Expectancy)', 'Economy (GDP per Capita)', data=east_asia, kind='kde', space=0, color='g')


# *Economy and Health seems to be highly correlated, so one of them should be eliminated in order to remove multicollinearity for prediction purposes.*

# * **Correlation for Middle East and Northern Africa**

# In[ ]:


plt.figure(figsize=(15,12))
middle_east = y_5.loc[lambda y_5 : y_5['Region'] == 'Middle East and Northern Africa']
sns.heatmap(middle_east.corr(), cmap = 'Blues', annot = True)
plt.show()


# *In Middle East and Northern Africa, Economy, Family and Freedom are the factors of uttermost importance(This explains the reason why Syria is in the worst 3 countries because the freedom is restricted and this country is under the control of ISIS).*

# * **Correlation for North America**

# In[ ]:


plt.figure(figsize=(15,12))
north_america = y_5.loc[lambda y_5 : y_5['Region'] == 'North America']
sns.heatmap(north_america.corr(), cmap = 'rainbow', annot = True)
plt.show()


# *Everything in this continent make the people happy and since USA is the most developed nation of the world, I am not surprised that North America is in the top 10 happy countries.*

# * **Correlation for Sub-Saharan Africa**

# In[ ]:


plt.figure(figsize=(15,12))
africa = y_5.loc[lambda y_5 : y_5['Region'] == 'Sub-Saharan Africa']
sns.heatmap(africa.corr(), cmap = 'Wistia', annot = True)
plt.show()


# *The most important factors for happiness in Sub-Saharan Africa is Family.*

# *Let's see top countries in each sector(features).*

# In[ ]:


# Getting top 10 countries in each sector
cols = ['Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity']

for col in cols:
    print(y_5[['Country', col]].sort_values(by = col,
    ascending = False).head(10))
    print("\n")


# # My Conclusion

# *What I concluded from this notebook is that all factors are equally important in maintaining happiness and peace among the people. No factor alone could decide the happiness among any nation's people.*

# **Hope you liked this notebook and learned something new. If you did, then please vote, it would mean a lot to me and let me know what can I do to improve myself in the comments. THANK YOU!!!!**

# In[ ]:




