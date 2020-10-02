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
import warnings
warnings.simplefilter(action='ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


missing_values=["n/a", "nan", "-"]
df=pd.read_csv('/kaggle/input/starbucks-menu/starbucks-menu-nutrition-drinks.csv', na_values=missing_values)


# In[ ]:


df.shape


# In[ ]:


df.head(15)


# In[ ]:


df_corr=df.corr()
f,ax=plt.subplots(figsize=(10,7))
sns.heatmap(df_corr, cmap='magma')
plt.title("Correlation between features", 
          weight='bold', 
          fontsize=18)
plt.show()


# A strong Correlation between:
# 
# 
# 
# Calories and Carbs
# 
# 
# Protein and Sodium
# 
# 
# 
# Calories and Proteins

# In[ ]:


df.isnull().mean().sort_values(ascending=False).plot.bar(color='black')
plt.axhline(y=0.1, color='r', linestyle='-')
plt.title('Missing values average per column', fontsize=20, weight='bold' )
plt.show()


# In[ ]:


df


# In[ ]:


df['Calories']=df['Calories'].fillna(method='ffill')
df['Fat (g)']=df['Fat (g)'].fillna(method='ffill')
df['Carb. (g)']=df['Carb. (g)'].fillna(method='ffill')
df['Fiber (g)']=df['Fiber (g)'].fillna(method='ffill')
df['Protein']=df['Protein'].fillna(method='ffill')
df['Sodium']=df['Sodium'].fillna(method='ffill')


# In[ ]:


NAcols=df.columns
for col in NAcols:
    if df[col].dtype != "object":
        df[col]= df[col].fillna(0)


# In[ ]:


df.isnull().sum().sort_values(ascending=False).head()


# In[ ]:



df[df['Unnamed: 0'].str.contains("Chocolate")]


# In[ ]:


df[df['Unnamed: 0'].str.contains("Vanilla")]


# In[ ]:


df[df['Unnamed: 0'].str.contains("Strawberry")]


# In[ ]:


df[df['Unnamed: 0'].str.contains("Lime")]


# In[ ]:


df[df['Unnamed: 0'].str.contains("Iced Coffee")]


# In[ ]:


df_filtered = df.query('Calories>100')
df_filtered


# In[ ]:


dffilt=df.query('Protein<30'and 'Sodium<25')
dffilt


# In[ ]:


df_filter = df[(df.Calories >= 100) & (df.Protein== 15)]
df_filter


# In[ ]:


df.filter(items=['Calories', 'Protein'])


# In[ ]:


df[df['Unnamed: 0'].str.contains('A|B')]


# In[ ]:


newdf = df[df.Protein.notnull()]
newdf


# Sorting by Name

# In[ ]:


df.sort_values("Unnamed: 0", axis = 0, ascending = True, 
                 inplace = True, na_position ='last') 
df


# In[ ]:


df.sort_values("Unnamed: 0", axis = 0, ascending = False, 
                 inplace = True, na_position ='last') 
df


# In[ ]:


df.sort_values("Unnamed: 0", axis = 0, ascending = True, 
                 inplace = False) 
df


# In[ ]:


df.sort_values("Sodium", axis = 0, ascending = False, 
                 inplace = True) 
df
  


# In[ ]:


df.sort_values(by=['Calories'], inplace=True, ascending=False)
df


# In[ ]:


df.sort_values(by=['Calories','Sodium'], inplace=True, ascending=False)
df


# In[ ]:


data=df.sort_values(by=['Calories'],ascending=False).sample(5)
c = data.style.background_gradient(cmap='Blues')
c


# In[ ]:


df.groupby(['Unnamed: 0']).mean().head()


# In[ ]:


df.groupby(['Unnamed: 0']).mean().sort_values('Calories', ascending=False).head()


# In[ ]:


df.groupby(['Unnamed: 0']).mean().sort_values(['Calories' and'Protein'], ascending=False).head()


# In[ ]:


Choco = df[(df.Calories >= 100) & (df.Protein== 15)]
Choco


# In[ ]:


df[['Calories',
      'Fat (g)',
      'Carb. (g)',
      'Fiber (g)',
      'Protein',
      'Sodium']].groupby(['Calories']).agg('median')


# In[ ]:


data['total_Calories'] = data['Fat (g)'] + data['Carb. (g)'] + data['Fiber (g)']+ data['Protein']+ data['Sodium']

sns.distplot(data['total_Calories'], color = 'magenta')

plt.title(' total Calories of coffees', fontweight = 30, fontsize = 20)
plt.xlabel('total calories')
plt.ylabel('count')
plt.show()


# From the graph above we can notice that the calories of the drinks most served in starbucks range between 140 and 210 calories.

# In[ ]:


plt.rcParams['figure.figsize'] = (50,50 )
plt.style.use('seaborn')
fig, ax = plt.subplots(figsize=(15,10))
sns.countplot(df['Calories'], palette = 'Greens')
plt.title('Calories count', fontweight = 100, fontsize = 20)
plt.xlabel('Calories')
plt.ylabel('Count')
plt.show()


# In[ ]:


sns.kdeplot(df['Calories'])


# Most of the coffees in starbucks are between 130 and 140 calories

# In[ ]:


print('Percentage',df.Calories.value_counts(normalize=True))
df.Calories.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


print('Percentage',df.Protein.value_counts(normalize=True))
df.Protein.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


print('Percentage',df.Sodium.value_counts(normalize=True))
df.Sodium.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


sns.barplot(x=df['Protein'], y=df['Calories'])


# In[ ]:


sns.barplot(x=df['Protein'], y=df['Sodium'])


# In[ ]:


sns.regplot(x=df['Protein'], y=df['Calories'])


# In[ ]:


sns.regplot(x=df['Protein'], y=df['Sodium'])


# In[ ]:


sns.regplot(x=df['Protein'], y=df['Carb. (g)'])


# In[ ]:


sns.regplot(x=df['Sodium'], y=df['Fiber (g)'])


# In[ ]:


sns.regplot(x=df['Fat (g)'], y=df['Fiber (g)'])


# In[ ]:


sns.regplot(x=df['Protein'], y=df['Carb. (g)'])


# In[ ]:


plt.figure(figsize=(15,15))
correlation=df.corr()
sns.heatmap(correlation,annot=True)


# In[ ]:


df


# In[ ]:


Lime= df[df['Unnamed: 0'].str.contains("Lime")]
Lime


# In[ ]:


sns.kdeplot(Lime['Calories'])


# In[ ]:



correlation=Lime.corr()
sns.heatmap(correlation,annot=True)


# In[ ]:


sns.regplot(x=Lime['Protein'], y=Lime['Carb. (g)'])


# In[ ]:


sns.regplot(x=Lime['Sodium'], y=Lime['Carb. (g)'])


# In[ ]:


sns.regplot(x=df['Calories'], y=df['Carb. (g)'])


# In[ ]:


sns.regplot(x=Lime['Protein'], y=Lime['Sodium'])


# In[ ]:


sns.countplot(x='Carb. (g)', hue='Calories', data=Lime)

