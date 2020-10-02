#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


def star(n):
    for i in range(0,n):
        for j in range(0,i+1):
            print("*", end="")
        print("\r")
n=5
star(n)


# In[ ]:


df=pd.DataFrame({'Team': ['India', 'Pakistan', 'Bangladesh', 'Australia','Peru'], 'Matches': [5,3,5,2,1],
                 'Runs':[1000, 500, 'null', 1100, 900]})
df


# In[ ]:


copied=df.copy()
copied


# In[ ]:


df['Runs'].replace('null',800, inplace=True)


# In[ ]:


df['Avg']=[20,25,40,45,50]


# In[ ]:


df['Result']=['win','loss','win','loss','win']


# In[ ]:


df


# In[ ]:


c=pd.get_dummies(df['Result'])
c


# In[ ]:


df1=df.join(c)


# In[ ]:


df1


# In[ ]:


df1.drop('Result', axis=1, inplace=True)


# In[ ]:


df1


# In[ ]:


bin=np.linspace(min(df1['Avg']), max(df1['Avg']),4)


# In[ ]:


groupnames=['low','medium','high']


# In[ ]:


df1['binned']=pd.cut(df1['Avg'], bin, labels=groupnames, include_lowest=True)


# In[ ]:


df1


# In[ ]:


df1.drop('binned', axis=1, inplace=True)


# In[ ]:


df1


# In[ ]:


df1['Team'].value_counts()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


plt.hist(df1['Runs'])
plt.xlabel('Team')
plt.ylabel('runs')
plt.title('Match Result')           


# In[ ]:


x=df1['Matches']
y=df1['Runs']


# In[ ]:


plt.scatter(x,y)


# In[ ]:


df1.groupby(df['Runs'])
df1


# In[ ]:


df1.sort_values('Matches', ascending=False, inplace=True)
df1


# In[ ]:


c=df1.corr()


# In[ ]:


c=df1.corr()
sns.heatmap(c, vmax=0.8, linewidth=0.5, annot=True, fmt='.1f', cmap='RdYlGn')


# In[ ]:


sns.regplot(x='Matches', y='Runs', data=df1)


# In[ ]:


from scipy import stats


# In[ ]:


t=pd.DataFrame({'House_Size': [2,4,5,7,8], 'Price': [100, 200, 300, 400, 500]})


# In[ ]:


from sklearn.linear_model import LinearRegression
ln=LinearRegression()
x=t['House_Size'].values
y=t['Price'].values
print(y)
print(x)


# In[ ]:


colors = ["red", "green", "blue", "purple"]
i = 0
while i < len(colors):
    print(colors[i])
    i += 1


# In[ ]:


fruits=['mango', 'apple', 'banana', 'orange']
i=0
while i < len(fruits):
    print(fruits[i])
    i+=1


# In[ ]:


fruits=['mango', 'apple', 'banana', 'orange']
i=0
for i in range(len(fruits)):
    print(fruits[i])
    i+=1


# In[ ]:


colors = ["red", "green", "blue", "purple"]
for color in colors:
    print(color)


# In[ ]:


presidents = ["Washington", "Adams", "Jefferson", "Madison", "Monroe", "Adams", "Jackson"]
for i in range(len(presidents)):
    print("President {}: {}".format(i+1 , presidents[i]))


# In[ ]:


presidents = ["Washington", "Adams", "Jefferson", "Madison", "Monroe", "Adams", "Jackson"]
for num, name in enumerate(presidents):
    print("President {}: {}".format(num+1, name))


# In[ ]:


colors = ["red", "green", "blue", "purple"]
ratios = [0.2, 0.3, 0.1, 0.4]
for i, color in enumerate(colors):
    ratio = ratios[i]
    print("{}% {}".format(ratio * 100, color))


# In[ ]:


colors = ["red", "green", "blue", "purple"]
ratios = [0.2, 0.3, 0.1, 0.4]
for color, ratio in zip(colors, ratios):
    print("{}% {}".format(ratio * 100, color))


# In[ ]:




