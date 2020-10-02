#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.shape


# **Getting to know the data**

# In[ ]:


train.head()


# In[ ]:


train.dtypes


# In[ ]:


#Categorical
sorted(train['Survived'].unique())


# In[ ]:


#Categorical
sorted(train['Pclass'].unique())


# In[ ]:


#Categorical
len(train['Name'].unique())


# In[ ]:


#Categorical
train['Sex'].unique()


# In[ ]:


#Numerical
train['Age'].describe()


# In[ ]:


#Numerical
train['SibSp'].describe()


# In[ ]:


#Numerical
train['Parch'].describe()


# In[ ]:


#TicketID
"Num of unique tickets = "+str(len(train['Ticket'].unique()))


# In[ ]:


#Numerical
train["Fare"].describe()


# In[ ]:


#CabinID
"Num of unique cabins = " + str(len(train['Cabin'].unique()))


# In[ ]:


#Categorical
train['Embarked'].unique()


# In[ ]:


#Cabin has a lot of nan values
train.isna().sum()


# In[ ]:


#Drop nan values and some of columns
train = train.drop(columns=['Ticket', 'Cabin', 'Name'], errors = 'ignore').dropna()
train.shape


# In[ ]:


print("Survived: {}%".format(train[train['Survived']!=0]["Age"].count()/train["Age"].count()*100))


# # **Encoding categorical values**
# Sex and Embarked have to be encoded

# In[ ]:


train["sex_encoded"] = train['Sex'].replace({'male':0,'female':1})
train["embarked_encoded"] = train["Embarked"].replace({'S':0,'C':1,'Q':2})
train.shape


# In[ ]:


#train = train.drop(columns = ["Sex", "Embarked"], errors = "ignore")
train.head(2)


# # **Covarience and correlation matrices. HeatMap**
# On the plot we can see, that there is positive relationship between columns Sex, Fare and Survived. And negative relationship between Pclass and Survived

# In[ ]:


train_cov = train.drop(columns = ["PassengerId"], errors = "ignore").cov()
train_cov


# In[ ]:


train_corr = train.drop(columns = ["PassengerId"], errors = "ignore").corr()
train_corr


# In[ ]:


fig = plt.figure(figsize=(10,7))
sns.heatmap(train_corr, annot = True)
fig.show()


# # **Analyzing Fare**
# * Dataset is normally distributed
# * Original dataset have outlier (>500). Need to drop outlier to normalize dataset.
# * Dataset is positevely skewed. 
# * Likelyhood of extreme evens is bigger, than in a normal distribution
# * Dataset have some extra peaks

# In[ ]:


fare = train[["Survived","Fare", "sex_encoded"]]


# In[ ]:


#Likelyhood of extreme events are very high 
fare['Fare'].kurtosis()


# In[ ]:


#Highly, positively skewed
fare['Fare'].skew()


# In[ ]:


#We definetely have outliers in a dataset :D
#Positively skewed
#We have some extra peaks around 25,55,80 values, need to dig deeper
fig = plt.figure(figsize=(10,8))
fig.title = ("KDE plot of Fare")
sns.distplot(fare['Fare'], rug = True, kde = True, hist = False)


# In[ ]:


sns.FacetGrid(fare, hue="Survived", height = 8).map(sns.distplot,"Fare")
plt.axvline(fare['Fare'].mean(), color = "g", label = 'mean')
plt.axvline(fare['Fare'].median(), color = "r", label = 'median')
plt.legend()


# In[ ]:


fig = plt.figure(figsize = (10,15))
sns.boxplot(x = "Survived", y = "Fare", data = train, hue = 'Sex')


# In[ ]:


fare_without_outliers = fare[fare['Fare']<200]
fare_without_outliers.describe()


# In[ ]:


#Positively skewed
fare_without_outliers['Fare'].skew()


# In[ ]:


#Extreme events likelyhood is slightly bigger, than a normal distribution without outlier
fare_without_outliers['Fare'].kurtosis()


# In[ ]:


#Distribution without outlier
fig = plt.figure(figsize=(10,8))
sns.distplot(fare_without_outliers['Fare'], rug = True, kde = True, hist = False)
plt.axvline(fare_without_outliers['Fare'].mean(), color = "g", label = 'mean')
plt.axvline(fare_without_outliers['Fare'].median(), color = "r", label = 'median')
plt.legend()


# In[ ]:


fig = plt.figure(figsize=(10,15))
sns.violinplot(x = "Survived", y = "Fare", data = fare_without_outliers, inner = None)
sns.swarmplot(x = "Survived", y = "Fare", data = fare_without_outliers, color = 'w')


# # **Analyzing Age**
# **After analizing plots, I made some predictions:**
# * People at the age of 0-15 could have more chances to survive;
# * People at the age of 15-50 had the smallest chances to survive.
# * People at the age of 50-80 had 50% to survive;
# 
# Biggest amount of survived were at the age of 20-50, because there were many more people at that age.
# 
# *There is no straight correlation in data["Age"], so we need to keep analyzing*

# In[ ]:


#Have extra pick
sns.distplot(train['Age'], rug=True,kde=True,hist=False)


# In[ ]:


sns.FacetGrid(train, hue="Sex").map(sns.distplot, 'Age').add_legend()


# In[ ]:


sns.boxplot(x="Survived", y="Age", hue="Sex", data=train)


# In[ ]:


fig, ax = plt.subplots(2, 2, figsize = (15,15))
fig.tight_layout(pad=2.5)
fig.suptitle("Age")
#taking all people's age and averaging. Sorting by index i.e. [0,1,...20,80]
data = train['Age'].value_counts().sort_index()
#survived
data_survived = train[train['Survived']!=0]["Age"].value_counts().sort_index()
data_dead = train[train['Survived']!=1]["Age"].value_counts().sort_index()

#Plot frequency chart of all ages
ax[0][0].bar(data.index, data.values)
ax[0][0].set_title("Frequency of each individual age")
ax[0][0].set_ylim([0,data.max()])
#Plot frequency chart of all survived 
ax[1][0].bar(data_dead.index, data_dead.values, label = "died")
ax[1][0].bar(data_survived.index, data_survived.values, label = 'survived')
ax[1][0].set_title("Survived over dead")
ax[1][0].legend()
ax[1][0].set_ylim([0,data.max()])
#Plot percentage of survivors in lines
ax[0][1].plot(data.index, data.values, label = "all")
ax[0][1].plot(data_survived.index, data_survived.values, label = "survived")
ax[0][1].set_title("Num of survivals at certain age")
ax[0][1].legend()
#Plot them all together
ax[1][1].bar(data.index, data.values, label = 'all')
ax[1][1].bar(data_survived.index, data_survived.values, label = 'survived')
ax[1][1].plot(data.index, data.values, label = "all",color = 'r')
ax[1][1].plot(data_survived.index, data_survived.values, label = "survived", color = 'b')
ax[1][1].set_xlim([0,train['Age'].max()])
ax[1][1].set_title("Putting all together")
ax[1][1].legend()

#d = (train[train['Survived']!=0]["Age"].dropna().value_counts().sort_index()/train["Age"].dropna().value_counts().sort_index())
#Filling gaps of None with mask
#s1mask = np.isfinite(d.values)
#ax[2][0].plot(d.index[s1mask], d.values[s1mask])
#ax[2][0].set_title("Percenile of survival at certain age")

#d2 = (train[train['Survived']!=1]["Age"].dropna().value_counts().sort_index()/train["Age"].dropna().value_counts().sort_index())
#s2mask = np.isfinite(d2.values)
#ax[2][1].plot(d2.index[s2mask], d2.values[s2mask], label = "dead")
#ax[2][1].plot(d.index[s1mask], d.values[s1mask], label = 'alive')
#ax[2][1].legend()
#ax[2][1].set_title("Percenile of death at certain age")


# # Lets see, if there are any dependencies in gender
# **After analizing first and second plots, I made strong prediction, that women had significantly more chances to survive!**

# In[ ]:


plt.figure(figsize=(5,5))
sns.countplot(train["Sex"])
plt.show()


# In[ ]:


plt.figure(figsize=(5,5))
sns.countplot(train["Sex"], hue = train["Survived"])
plt.show()


# In[ ]:


gender_and_age = train[["Survived", "Sex", "Age"]].replace({'female':0, 'male':1}).round(0).groupby(['Age', 'Sex', 'Survived']).size()
colors = {0:'r', 1:'b'}
titles = {0:'female', 1:'male'}
fg,ax = plt.subplots(1,2)
fg.suptitle("Female and male (red - female, blue - male)")
ax[0].set_title("Survived")

ax[1].set_title("Died")
for age, sex, survived in gender_and_age.index:
    num = gender_and_age[age][sex][survived]
    colored = colors[sex]
    titled = titles[sex]
    if survived:
        ax[0].scatter(age, num, color = colored)
    else:
        ax[1].scatter(age, num, color = colored)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


x_train = train.drop(columns=["PassengerId", "Survived"])
y_train = train[["PassengerId", "Survived"]]


# In[ ]:




