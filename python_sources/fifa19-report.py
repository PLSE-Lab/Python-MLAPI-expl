#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/data.csv')


# In[ ]:


df.info()


# In[ ]:


columns = ['Value','ID','Age','Nationality','Overall','Potential','Wage','Preferred Foot']
data = df[columns]


# In[ ]:


data = data.dropna()


# In[ ]:


data['Preferred Foot'].unique()


# In[ ]:


def foot(x):
    if x=='Left':
        return 0
    else:
        return 1
data['Pre Foot'] = data['Preferred Foot'].apply(foot)


# In[ ]:


data['Pre Foot'].head()


# In[ ]:


def value(x):
    if 'M' in x:
        return float(x[1:-1])*1000
    elif 'K' in x:
        return float(x[1:-1])
    else:
        return 0
data['value'] = data['Value'].apply(value)


# In[ ]:


data['wage'] = data['Wage'].apply(value)


# In[ ]:


data.describe()


# In[ ]:


sns.distplot(data.value)
print(data[data.value>10000]['value'].count())


# The value is mainly concentrated around 600, high-value players account for a very small part, and above 10,000K, there are only 879 players, accounting for less than 5%.

# In[ ]:


sns.pairplot(data.drop('ID',axis=1))


# In[ ]:


plt.figure(figsize=(10,8))
corr = data.drop('ID',1).corr()
sns.heatmap(corr,annot=True)


# From the picture,the value of the player is strongly correlated with current wages, generally related to overall performance and potential, and not linearly related to age.

# Next, analyze one by one

# ## Preferred Foot

# In[ ]:


sns.countplot(data['Preferred Foot'])
print(data['Pre Foot'].sum()/data['Pre Foot'].count())


# In[ ]:


sns.violinplot(y = 'value',x='Preferred Foot',data=data[data.value<10000])
plt.figure()
sns.stripplot(x='Preferred Foot',y='value',data=data)


# 76.8% of players are accustomed to using their right foot. But the habit of using the foot is also irrelevant to the value of the player.

# ## Nationality

# In[ ]:


plt.figure(figsize=(30,8))
sns.countplot(data['Nationality'])
plt.xticks(rotation=90)
print(data['Nationality'].value_counts().sort_values(ascending=False).index[:5].values)


# In[ ]:


sns.countplot(x='Nationality',order=['England','Germany','Spain','Argentina','France'],data=data)
plt.figure()
sns.violinplot(x='Nationality',y='value',data=data[data['value']<10000],order=['England','Germany','Spain','Argentina','France'])
plt.title('value<10000')
plt.figure()
sns.violinplot(x='Nationality',y='value',data=data[data['value']<5000],order=['England','Germany','Spain','Argentina','France'])
plt.title('value<50000')
plt.figure()
sns.violinplot(x='Nationality',y='value',data=data[data['value']<2000],order=['England','Germany','Spain','Argentina','France'])
plt.title('value<20000')


# England' 'Germany' 'Spain' 'Argentina' 'France' are the most players in these countries. Selected players from these five countries for analysis, found that players in different countries, the player value distribution is slightly different, such as England and France in comparison, when the value is below 2000, France's value distribution is more flat, and England's players are more concentrated.

# ## wage

# In[ ]:


sns.distplot(data['wage'])


# In[ ]:


sns.regplot(x='wage',y='value',data=data,marker='.')


# The distribution characteristics of the overall player salary are similar to the value. Most of the wages are also concentrated, and there are a few players with high wages. It can be seen that there is a clear correlation between value and wages. The higher the value, the higher the salary, and there is no player who value to be buried.

# ## Age

# In[ ]:


sns.distplot(data['Age'],kde=False,color='deepskyblue')


# In[ ]:


sns.regplot(x='Age',y='value',data=data,fit_reg=False,color='darkgrey')


# It can be seen that there is no linear relationship between age and player value, but the value of valuable players is between 20-30. Because the age is too small, there is no practical experience, and the age can hardly keep up.

# ## Overall

# In[ ]:


sns.boxplot(y='Overall',data=data,width=0.4,color='lime')


# In[ ]:


def log(x):
    return np.log10(x)
test=data[['Overall','value']].copy()
test['logvalue'] = test['value'].apply(log)


# In[ ]:


sns.lmplot('Overall','logvalue',data=test)


# In[ ]:


test.corr()


# The overall performance of the players is mainly between 60-70. The overall performance of the player has a strong correlation with the logarithm of the player's value, with a correlation coefficient of 0.938, so it has a big impact.

# ## Potential

# In[ ]:


sns.boxplot(y = 'Potential',data=data,width=0.4)


# In[ ]:


data.plot.scatter('Potential','value',logy=True)
plt.ylim(1,10**6)


# Similar to the overall performance, the player's potential and value logarithm are also strongly correlated.

# ## conclusion

# The value is mainly concentrated around 600, and high-value players account for a very small part. High-value players are not expected.

# The player's overall performance and potential have a strong correlation with the player's logarithm and are the main influencing factors of player value. There is a strong correlation between the player's salary and the player's value.

# Age has a certain non-linear relationship with the value of players. High-value players are concentrated between 20-30.

# The habits of the country and the feet have little effect on the value of the players.
