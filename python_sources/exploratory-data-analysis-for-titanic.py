#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# 

# This notebook is an example of loading datasets and checking out multiple variables.

# ## Data Loading
# First import python packages.

# In[24]:


import pandas as pd
import numpy as np


# In[25]:


train = pd.read_csv('../input/train.csv')


# After loading the data, let's first check all the variable names and their variable types.

# In[26]:


train.columns.tolist()


# Check the variable types using the `info` function of pandas dataframe.

# In[27]:


train.info()


# Take a glance of the first 10 observations.

# In[28]:


train.head(n=10)


# Some variables are not useful for our task, such as `PassengerId`, `Name`, `Ticket`, `Cabin` and `Fare`. Therefore I can remove these four columns.

# In[29]:


columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare']
train.drop(columns, inplace=True, axis=1)


# We also notice that some categorical variables were identified as integers or objects, such as `Pclass`, `Sex` and `Embarked`. We need to coerce their data types. The `Survived` column had better be transformed into a boolean variable.

# In[30]:


columns = ['Pclass', 'Sex', 'Embarked']
for column in columns:
    train[column] = train[column].astype('category')


# In[31]:


train['Survived'] = train['Survived'].astype('bool')


# ## Summary Statistics and Data Visualization

# First import some packages corresponding to plotting.

# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[33]:


train.head()


# An easy and straight forward way is to use the `describe` function. It will reflect the basic statistics of numeric variables.

# In[34]:


train.describe()


# The target variable is survival rate. In the training dataset, the survival rate is 38.4%.

# In[35]:


train['Survived'].mean()


# Then we look at every indicator variable one by one. Start from `Pclass`.

# In[36]:


train.groupby('Pclass')['Survived'].mean()


# Survival rate is highest for passenger class 1(62.9%). Passenger class 3 has the smallest survival rate(24.2%). This agrees with history.

# Next we look at `Sex`.

# In[37]:


nummale = train[train['Sex']=='male']['Sex'].count()
numfemale = len(train)-nummale
print('There are', nummale,'male passengers and', numfemale, 'female passengers')


# In[38]:


train.groupby('Sex')['Survived'].mean()


# The survival rate of women is much higher than men. This is probably because the lifeboats were given away to women first.

# Then it comes to `Age`. First let's use histograms to see the distribution of all the people.

# In[39]:


sns.set()
train.hist(column='Age')
plt.xlabel('Age')
plt.title('')
plt.show()


# In[40]:


train.hist(column='Age', by='Survived', sharey=True)
plt.show()


# In[41]:


train.hist(column='Age', by='Sex', sharey=True)
plt.show()


# In[42]:


train.hist(column='Age', by='Pclass', sharey=True)
plt.show()


# Through the histograms above, we can see that male passengers aging from 20 to 30 suffer from much higher casualties than other age groups.

# We can use the same approach to peek into `SibSp` and `Parch`.

# In[43]:


train.hist(column='SibSp')
plt.show()


# In[44]:


train.hist(column='Parch')
plt.show()


# Since most of the passengers don't have siblings or parents or children, we can set these variables to binary variables(has or hasn't) for our future model training.

# In[45]:


train['withsibsp'] = train['SibSp'] > 0
train['withparch'] = train['Parch'] > 0


# In[46]:


train.groupby('withsibsp')['Survived'].mean()


# In[47]:


train.groupby('withparch')['Survived'].mean()


# Passengers with company seem to have higher survival rates.

# Finally let's look at the `Embarked` column.

# In[48]:


train.groupby('Embarked')['Embarked'].count()


# In[49]:


train.groupby('Embarked')['Survived'].mean()


# I assumed that where the passenger aboard the ship doesn't matter, but those from Cherbrough had a much higher survival rate than the other two.

# ## Summary
# This elementary data summary and visualization have revealed some trends. Age, sex and passenger class are useful indicators of survival rate. The next step is to use statistical models to quantify the effect of individual indicators and interactions between them.
