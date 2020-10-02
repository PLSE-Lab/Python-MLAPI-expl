#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# **Credit to :  Helge Bjorland & Stian Eide for they tutorial which helped me **

# In[ ]:




train_csv = pd.read_csv('../input/train.csv')
test_csv = pd.read_csv('../input/test.csv')

full = train_csv.append( test_csv , ignore_index = True )


train = full[:891]
test= full[891:]

print ('Datasets:' , 'full:' , full.shape , 'titanic:' , full.shape)


# In[ ]:


test.head(10)


# In[ ]:


train.head(30)


# _____
# ## **1) Data Exploring**

# In[ ]:


train.shape


# In[ ]:


train.dtypes


# **Categorical:**  Survived, Sex, and Embarked. Ordinal: Pclass.
# 
# **Continous:**  Age, Fare. Discrete: SibSp, Parch.

# 

# In[ ]:


full.isnull().any().count()


# **Blank values ?**
# 
# Cabin, Age, Embarked features contain a number of null values.

# In[ ]:


train.describe()


# _____
# ## **2) Data Exploring**

# In[ ]:


fig, ax = plt.subplots(figsize=(8,8)) 
ax1 = sns.heatmap(full.isnull(), cbar=False)


# In[ ]:


msno.heatmap(train)


# In[ ]:


null_columns = train.columns[train.isnull().any()]
train[null_columns].isnull().sum()


# In[ ]:


full = full.drop(labels='Cabin',  axis=1)


# In[ ]:


full.head()


# In[ ]:


full.groupby(['Sex'])['Sex'].count().plot.pie()


# In[ ]:


full.groupby(['Embarked'])['Embarked'].count().plot.pie()


# In[ ]:


full.groupby(['Pclass'])['Pclass'].count().plot.pie()


# In[ ]:


train.groupby(['Survived'])['Survived'].count().plot.pie()


# In[ ]:


full.groupby(['Parch'])['Parch'].count().plot.pie()


# In[ ]:


full_age = full['Age'].dropna().astype(int)
f, axes = plt.subplots( figsize=(7, 7), sharex=True)
sns.despine(left=True)

sns.distplot(full_age, color="b", bins=40)


# In[ ]:


#survived = 'survived'
#not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))


women = full[full['Sex']=='female']
men = full[full['Sex']=='male']


ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=20, label ='survived', ax = axes[0], kde=False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label ='not survived', ax = axes[0], kde=False)
ax.legend()
ax.set_title('Female')


ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=20, label = 'survived', ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = 'not survived', ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')


# ______

#  ## **Category variable to num**

# In[ ]:


full['Age'].fillna(full['Age'].mean(), inplace=True)


# In[ ]:


sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) )
full.Sex = sex


# In[ ]:


full.Sex.head()


# _______
# 

# In[ ]:


names = full['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
names 


# In[ ]:


titles = pd.get_dummies( names )
titles.head()


# In[ ]:


embarked = pd.get_dummies( full.Embarked , prefix='Titl' )
embarked.head()


# In[ ]:


embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
embarked.head()


# In[ ]:


pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )
pclass.head()


# In[ ]:


full.isnull().sum()


# In[ ]:


full['Fare'].fillna(full['Fare'].mean(), inplace=True)


# In[ ]:


full['Embarked'].fillna(1, inplace=True)
full.isnull().sum()
#full.head()
#train = train.dropna()


# In[ ]:


corr = train.corr()
f , ax = plt.subplots( figsize =( 12 , 10 ) )
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
f = sns.heatmap(corr, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize':12})


# _____
# 

# # Modeling

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from learntools.core import *

train.head()


# In[ ]:


y = train.Survived


# In[ ]:


full_X = pd.concat([embarked, pclass, titles, full['Age'], sex, full['Fare'], full['SibSp']], axis=1 )
full_X.head()


# In[ ]:


train_Xval = full_X[:891]
test_X = full_X[891:]


# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(train_Xval, y, random_state=1, train_size =.8)


# In[ ]:


#d_model = DecisionTreeClassifier(random_state=1)
d_model = RandomForestClassifier(random_state=1)


# In[ ]:


d_model.fit(train_X, train_y)


# In[ ]:


val_predictions = d_model.predict(val_X)
val_predictions


# In[ ]:


print (d_model.score( train_X , train_y ) ,d_model.score( val_X , val_y ))


# In[ ]:


test_Y = d_model.predict( test_X )
#test_Y.astype(int)


# In[ ]:


passenger_id = full[891:].PassengerId
passenger_id.head()


# In[ ]:


test_Y.shape


# In[ ]:


final = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )


# In[ ]:


final = final.astype(int)
final.head()


# In[ ]:


final.to_csv( 'titanic_pred.csv' , index = False )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




