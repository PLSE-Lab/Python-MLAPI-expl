#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# I've been dealing with taitanic data while learning data analysis, but it's my first time in kaggle. 
# Accuracy was about 77% before, so let's aim to get more than that score.
# First of all, I will do data grasping and EDA in this notebook.
# 
# # Load packages and datasets

# In[ ]:


import numpy as np 
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
print("train shape : ", train.shape)
print("test shape : ", test.shape)


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# # EDA
# 
# ## target value

# In[ ]:


train["Survived"].value_counts().plot(kind="pie",explode=[0,0.05],autopct='%1.1f%%')
plt.title("Ratio of Survived",weight = "bold")


# ## Missing Value

# In[ ]:


train1 = train[train.columns[train.isnull().sum()!=0]]
test1 = test[test.columns[test.isnull().sum()!=0]]
na_prop1 = (train1.isnull().sum()).sort_values()
na_prop2 = (test1.isnull().sum()).sort_values()
plt.figure(figsize=(10,8))
sns.set_style('whitegrid')
plt.subplot(211)
na_prop1.plot.barh(color='blue')
plt.title('Missing values(train set)', weight='bold')
plt.subplot(212)
na_prop2.plot.barh(color='blue')
plt.title('Missing values(test set)', weight='bold' )


# Cabin has many missing values.
# 
# ## Observation of variables

# In[ ]:


train1 = train.select_dtypes(include = "object").drop(["Name","Ticket","Cabin"],axis=1)
plt.figure(figsize=(12,6))
plt.style.use('ggplot')
for i, col in enumerate(train1.columns):
    plt.subplot(1,2,i+1)
    sns.countplot(col,hue = "Survived",data = train)


# People male and embarked with S has low survival rate.

# In[ ]:


train1 = train.select_dtypes(include = "int").drop(["PassengerId","Survived"],axis=1)
plt.figure(figsize=(12,5))
plt.style.use('ggplot')
for i, col in enumerate(train1.columns):
    plt.subplot(1,3,i+1)
    sns.countplot(col,hue = "Survived",data = train)


# People SibSp and Parch with 0 and Pclass with 3 has low survival rate.

# In[ ]:


train1 = train.select_dtypes(include = "float")
plt.figure(figsize=(12,5))
plt.style.use('ggplot')
for i, col in enumerate(train1.columns):
    plt.subplot(1,2,i+1)
    sns.distplot(train[train["Survived"]==0][col],color='red')
    sns.distplot(train[train["Survived"]==1][col],color='blue')


# people have fare clear to 0 has a lower survival rate. And shows people younger than 5 has a higher survival rate.
# I'm going to divide the age into some groups.
# Create new variable by adding SibSp and Parch.

# In[ ]:


plt.figure(figsize=(12,5))
train["Family"] = train["SibSp"] + train["Parch"] + 1
sns.countplot("Family",hue="Survived",data= train)


# In[ ]:


train['Age_group'] = pd.qcut(train['Age'], 10)
plt.figure(figsize=(14,7))
sns.countplot(x='Age_group', hue='Survived', data=train)


# ## Cabin
# 
# Cabin has many missing value, but once deal with rest values.

# In[ ]:


plt.figure(figsize=(14,5))
train["Cabin"].fillna("None",inplace=True)
train["Cabin_First"] = [i[0][0] for i in train["Cabin"]]
plt.subplot(121)
sns.barplot("Cabin_First","Survived",data = train)
plt.title("Cabin First vs Survived")
plt.subplot(122)
sns.countplot(train[train["Cabin_First"]!="N"]["Cabin_First"])
plt.title("Cabin First Count(without NA value)")


# Cabin vs Pclass percentage

# In[ ]:


cabin_pcl_group = train.groupby(["Cabin_First","Pclass"])["PassengerId"].count()
cabin_pcl_pers = {}
for i in train["Cabin_First"].unique():
    cabin_pcl_pers[i] = cabin_pcl_group[i] / cabin_pcl_group[i].sum()
    for pcl in range(1,4):
        try:
            cabin_pcl_pers[i][pcl]
        except:
            cabin_pcl_pers[i][pcl] = 0
cabin_pcl_pers = pd.DataFrame(cabin_pcl_pers).T

plt.figure(figsize = (11,5))
plt.bar(np.arange(9),cabin_pcl_pers[1], color='skyblue', width=0.7, label='Pclass = 1')
plt.bar(np.arange(9),cabin_pcl_pers[2], bottom=cabin_pcl_pers[1], color='lightpink', width=0.7, label='Pclass = 2')
plt.bar(np.arange(9),cabin_pcl_pers[3], bottom=cabin_pcl_pers[1] + cabin_pcl_pers[2], color='lightgray', width=0.7, label='Pclass = 3')
plt.xticks(np.arange(9),cabin_pcl_pers.index)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Percentage of Pclass with Cabin')


# ## Name
# After looking at the Name variables's shape, only the first letter was taken.

# In[ ]:


train['Name'].head()


# In[ ]:


train['First_Name'] = train['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
train['First_Name'].value_counts()


# Those with low frequency were grouped together.

# In[ ]:


train['First_Name_label'] = [0 if name=='Mr' else 1 if name=='Mrs'  else 2 if name=='Miss' else 3 if name=='Master' else 4 for name in train['First_Name']]
plt.figure(figsize=(8,5))
sns.barplot(x='First_Name_label', y='Survived', data=train)
plt.xticks(np.arange(5),['Mr','Mrs','Miss','Master','Other'])


# # Ticket
# 
# Ticket variable have different numbers and different shapes. So the frequency of those with the same number was measured

# In[ ]:


print(train["Ticket"].head(10))
print("Ticket unique value number : " , train["Ticket"].nunique())


# In[ ]:


ticket_group = train.groupby('Ticket')['PassengerId'].count().reset_index()
ticket_group.columns = ['Ticket',"Ticket_freq"]
train_t = train.merge(ticket_group, on = "Ticket")
plt.figure(figsize=(13,10))
plt.subplot(211)
sns.countplot(x='Ticket_freq', hue='Survived', data=train_t)
plt.subplot(212)
sns.barplot(x='Ticket_freq', y='Survived', data=train_t)


# ## Survial rate by Sex and Pclass

# In[ ]:


train_group = train.groupby(['Pclass','Sex'])['Survived'].mean()

plt.figure(figsize=(10,7))
sns.heatmap(train_group.unstack("Pclass"), cmap='Blues', annot=True)


# ## Survial rate by Embarked and Pclass

# In[ ]:


train_group = train.groupby(['Pclass','Embarked'])['Survived'].mean()
plt.figure(figsize=(10,7))
sns.heatmap(train_group.unstack("Pclass"), cmap='Blues', annot=True)


# ## Survial rate by Sex, Pclass and Embarked

# In[ ]:


train.groupby(['Sex','Pclass','Embarked'])['Survived'].mean()


# When I saw these, I think the influence of Embarked does not seem to be great. Let's do a more detailed analysis

# In[ ]:


sns.factorplot('Embarked','Survived',hue='Pclass',data=train)


# ## chi square test

# In[ ]:


from scipy.stats import chi2_contingency

df1 = train.groupby(["Embarked","Survived"])["Survived"].count().unstack("Survived")
chi2, p, dof, expected = chi2_contingency(df1)
msg = 'Embarked Statistic: {}\np-value: {}\nDegree of Freedom: {}'
print(msg.format(chi2, p, dof))

df2 = train.groupby(["Pclass","Survived"])["Survived"].count().unstack("Survived")
chi2, p, dof, expected = chi2_contingency(df2)
msg = 'Pclass Statistic: {}\np-value: {}\nDegree of Freedom: {}'
print(msg.format(chi2, p, dof))

df3 = train.groupby(["Sex","Survived"])["Survived"].count().unstack("Survived")
chi2, p, dof, expected = chi2_contingency(df3)
msg = 'Sex Statistic: {}\np-value: {}\nDegree of Freedom: {}'
print(msg.format(chi2, p, dof))


# Embarked's statistics are small compared to other variables. but it can be said that there is a difference.

# ## correlation

# In[ ]:


plt.figure(figsize = (8,8))
sns.heatmap(train.drop("PassengerId",axis=1).corr(), cmap = "Blues", annot = True)


# Next notebook, I wiil do data preprocessing, modeling, and further work
# 
# 
# ## Next
# https://www.kaggle.com/rbud613/taitanic-predictor-using-pipeline-processing
