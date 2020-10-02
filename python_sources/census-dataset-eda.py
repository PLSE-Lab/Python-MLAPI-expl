#!/usr/bin/env python
# coding: utf-8

# # EDA census data of income

# The goal of this project is to predict 
# whether a person makes over 50K a year or not
# given their demographic variation. This is a classification problem.

# ### Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# ### Read dataset

# In[ ]:


censusData = pd.read_csv('../input/adult.csv')
censusData.head()


# In[ ]:


censusData.shape


# #### Convert empty data to standard null values

# In[ ]:


censusData.replace({'?':np.nan},inplace=True)


# #### Analyse null values

# In[ ]:


censusData.info()


# In[ ]:


censusData.isnull().sum()


# # Statistical Summary

# In[ ]:


censusData.describe()   


# #### Summary of categorial values

# In[ ]:


censusData.describe(include=np.object)


# #### Fill null values

# In[ ]:


censusData['workclass'].fillna(censusData['workclass'].mode().values[0],inplace=True)
censusData['occupation'].fillna(censusData['occupation'].mode().values[0],inplace=True)
censusData['native-country'].fillna(censusData['native-country'].mode().values[0],inplace=True)


# In[ ]:


censusData.info()


# In[ ]:


censusData.head()


# ## Univariate Analysis
# #### Numerical data

# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(censusData['age'],kde=False)
plt.show()


# In[ ]:


(censusData.age>70).sum()       # right skewed but data > 70 is less person so not an issue 


# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(censusData['hours-per-week'],kde=False,color='r')
plt.show()

''' 
    3rd quantile is 45 i.e 75% spend less than 45 hr/week.
    majority population spend b/w 40-45 hr/wk
    '''


# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(censusData['capital-gain'],kde=False,color='b',bins=15)
plt.show()

''' it shows either no gain or high gain due to high deviation'''


# In[ ]:


print(censusData[['capital-gain','capital-loss']].corr())

plt.figure(figsize=(8,4))
sns.scatterplot(censusData['capital-gain'],censusData['capital-loss'])
plt.show()

'''
capital gain and loss have negative weak relationship.
If one is 0 ,another is high
'''


# ### Categorial Data

# In[ ]:


plt.figure(figsize=(15,6))

sns.countplot(censusData['workclass'])
plt.show()

'''
most people working in private sector.
Huge imbalance in data.
'''


# In[ ]:


plt.figure(figsize=(18,10))

ax = sns.countplot(censusData['education'])

for p in ax.patches:
    
    x = p.get_x()+p.get_width()/2 - 0.1
    y = p.get_height() + 3
    t = round((p.get_height()/censusData.shape[0])*100)
    
    ax.text(x,y,t)
    
plt.show()

'''
most people are HS-grad(32%) followed by some college and bachelors.

'''


# In[ ]:


sns.countplot(censusData['marital-status'])
plt.xticks(rotation = 90)
plt.show()

'''
majority is married with civilian spouse having highest followed by never married.
'''


# In[ ]:


sns.countplot(censusData['occupation'])
plt.xticks(rotation = 90)
plt.show()

'''
Prof-specialty has the maximum count(8981) but Craft-repair, Exec-managerial and Adm-clerical Sales has comparable number of observations.
Armed-Forces has minimum samples in the occupation attribute.
'''


# In[ ]:


sns.countplot(censusData['gender'])
plt.show()
'''
males are in majority
'''


# In[ ]:


sns.countplot(censusData['relationship'])
plt.xticks(rotation = 90)
plt.show()

'''
husband have highest frequency
'''


# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot(censusData['native-country'])
plt.xticks(rotation = 90)
plt.show()

'''
US have highest frequency, that means data is of country US.
'''


# # Label Encoding for transformation

# In[ ]:


censusData['income'] = LabelEncoder().fit_transform((censusData['income']))


# In[ ]:


censusData.head()


# In[ ]:


sns.countplot(censusData['income'])
plt.show()
'''
people with income less than 50k are higher.
'''


# # Bivariate analysis
# 
# numerical vs catogorical

# In[ ]:


sns.boxplot(x='income',y='age',data=censusData)
plt.show()
'''
avg people's age is 44 approx. for income > 50k which is more compared to income <= 50k
'''


# In[ ]:


plt.figure(figsize=(12,5))
sns.boxplot(x='income',y='hours-per-week',data=censusData)
plt.show()

''' income >50k have longer avg working hrs than <50k
working hr range is also higher'''


# In[ ]:


plt.figure(figsize=(12,5))
sns.boxplot(x='income',y='fnlwgt',data=censusData)
plt.show()

''' weight has no significance on income'''


# ### Categorial vs Categorial

# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot(x='workclass',hue='income',data=censusData)
plt.show()

''' private working class have highest no. of people >50k
however conversion ratio is minimal in private while in self-emp-inc have very high conversion ratio'''


# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot(x='education',hue='income',data=censusData)
plt.xticks(rotation = 90)
plt.show()

''' higher education have better chance of earning >50k'''


# In[ ]:


plt.figure(figsize=(8,4))
sns.countplot(x='gender',hue='income',data=censusData)
plt.xticks(rotation = 90)
plt.show()

''' females are more likely to be earning <= 50k'''


# ### Stacked bar for catorgorical vs catogorical

# In[ ]:


dd = pd.crosstab(censusData['occupation'],censusData['income'])
dd.plot.bar(stacked=True,figsize=(12,5))
plt.xticks(rotation = 90)
plt.show()

''' In every occupation, people who earn less than 50k is greater than people who earn >50k.
'''


# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot(x='native-country',hue='income',data=censusData)
plt.xticks(rotation = 90)
plt.show()

'''country has negligible effect on income'''


# ### Multivariate analysis

# In[ ]:


plt.figure(figsize=(12,5))
sns.boxplot(censusData['income'],censusData['hours-per-week'],hue=censusData['gender'])
plt.show()
'''average working hour of female is less compared to males.
However for income group >50k mens have more flexible
working hrs compared to females. '''


# In[ ]:


plt.figure(figsize=(12,5))
sns.boxplot(censusData['income'],censusData['age'],hue=censusData['gender'])
plt.show()
'''average age of female is less compared to males for income group >50k.
However range is less for men compared to females for group <=50k. '''


# In[ ]:


plt.figure(figsize=(12,5))
sns.boxplot(censusData['workclass'],censusData['age'],hue=censusData['gender'])
plt.show()
'''average age of private and gov dept is less for income group >50k compared to group <=50k. '''


# ### Feature Engineering

# In[ ]:


censusData['capital_range'] = censusData['capital-gain'] - censusData['capital-loss']
censusData.head()


# In[ ]:


sns.distplot(censusData['capital_range'],kde=False)
plt.show()

'''capital range has same plot as capital gain/loss so we can replace capital gain and loss with one column.'''


# ### Correlaton using Heatmap

# In[ ]:


plt.figure(figsize=(12,5))
sns.heatmap(censusData.corr(),annot=True)
plt.show()


# ## Conclusion
1.People with income less than 50k are higher.
2.Average people's age is 44(approx.) for income > 50k which is more compared to income <= 50k.
3.Weight has no significance on income.
4.Private working class have highest no. of people >50k.However conversion ratio is minimal in private while in     self-emp-inc have very high conversion ratio.
5.Higher education have better chance of earning >50k.
6.Females are more likely to be earning <= 50k.
7.In every occupation, people who earn less than 50k is greater than people who earn >50k.
8.Average working hour of female is less compared to males.
However for income group >50k mens have more flexible
working hrs compared to females.
9.Average age of private and gov dept is less for income group >50k compared to group <=50k.