#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings


warnings.filterwarnings("ignore")


# In[ ]:


get_ipython().system('ls ../input/titanic')


# In[ ]:


df_train = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
df_test = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')


# In[ ]:


df_train.head(2)


# In[ ]:


df_train.shape


# In[ ]:


df_test.head(2)


# In[ ]:


df_test.shape


# Train set has 891 records.
# Test set has 418 records.

# # overview over columns
# ## Pclass

# In[ ]:


df_train['Pclass'].unique()


# In[ ]:


df_train['Pclass'].value_counts().sort_index().plot.bar(grid=True)
plt.title('Number of people in each class')
plt.show()


# In[ ]:


df_train.groupby(['Pclass'])['Survived'].mean()


# In[ ]:


df_train.groupby(['Pclass', 'Survived'])['Survived'].count().unstack(1).plot.bar(grid=True)
plt.title('The number of people in each class divided into survival')
plt.show()


# Depands on class most passangers survived from class 1 (`62%`).

# ## Sex

# In[ ]:


df_train['Sex'].unique()


# In[ ]:


df_train['Sex'].value_counts().sort_index().plot.bar(grid=True)
plt.title('The number of people by gender')
plt.show()


# In[ ]:


df_train.groupby(['Sex', 'Survived'])['Survived'].count().unstack(1).plot.bar(grid=True)
plt.title('The number of people by gender divided into survival')
plt.show()


# In[ ]:


df_train.groupby(['Sex', 'Pclass'])['Survived'].mean()


# In[ ]:


df_train.groupby(['Sex', 'Survived', 'Pclass'])['Survived'].count().unstack(1).plot.bar(grid=True)
plt.title('The number of people by gender divided into survival and Pclass')
plt.show()


# Woman has higher chance for survive.
# 
# Most of womane from class 1 `97%` and 2`92%` survived. Woman in class 3 have `50%` chance for survive.
# 
# Man has lower change for survive. Class 1 - `36%`, class 2 - `16%` and class 3 - 1`4%`.

# ## Name

# In[ ]:


titles = [
    'Mr.', 'Mrs.', 'Miss.', 'Master.', 'Don.', 
    'Rev.', 'Dr.', 'Mme.', 'Ms.', 'Major.', 
    'Lady.', 'Sir.', 'Mlle.', 'Col.', 'Capt.',
    'Countess.', 'Jonkheer.'
]


#  Title list.
#  I checked it manually. I created list with all titles in our training data set.

# In[ ]:


df_train['Name'][~df_train['Name'].apply(lambda x: any(title in x for title in titles))]


# In[ ]:


for title in titles:
    df_train[title] = df_train['Name'].apply(lambda x: title in x)


# In[ ]:


df_train[titles].sum()


# There is a lot of persons with single title occurs.
# I decided to join them into one column `others`.

# In[ ]:


df_train.drop(titles, axis='columns', inplace=True)

updated_titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.']

for title in updated_titles:
    df_train[title] = df_train['Name'].apply(lambda x: title in x)
    
df_train['others'] = ~df_train[updated_titles].any(axis='columns')


# In[ ]:


df_train.head(2)


# Created new features. They contain information about person title. I separated 5 groups: `Mr.`, `Mrs.`, `Miss.`, `Master.` and `others`. Last group others contain all people with diffrent title.

# In[ ]:


df_train.groupby(['others']+updated_titles)['Survived'].mean()


# In[ ]:


df_train.groupby(['others']+updated_titles)['Survived'].count()


# In[ ]:


ax = df_train.groupby(['others', 'Survived']+updated_titles)['Survived'].count().unstack(1).plot.bar(grid=True)
ax.set_xticklabels((['others']+updated_titles)[::-1])
# plt.yscale('log')
plt.title('The number of people by title divided into survival')
plt.show()


# Title `Mr.` was most deadly (16% allive). `Miss.` and `Mrs` was most freqent titles in survivels.

# ## Age

# In[ ]:


df_train['Age'].describe()


# In[ ]:


df_train['Age'].plot.hist(grid=True)
plt.title('The number of people by age')
plt.show()


# Most of people on a Titanic was young. Mean is 30 years old. Median is 28.

# In[ ]:


plt.hist(
    [
        df_train['Age'][df_train['Pclass']==1], 
        df_train['Age'][df_train['Pclass']==2], 
        df_train['Age'][df_train['Pclass']==3]
    ],
    bins=15, stacked=True)
plt.legend(['Pclass 1', 'Pclass 2', 'Pclass 3'])
plt.title('The number of people by age divided into Pclass')
plt.grid()
plt.show()


# In[ ]:


df_train['Age'][df_train['Pclass']==1].describe()


# In[ ]:


df_train['Age'][df_train['Pclass']==2].describe()


# In[ ]:


df_train['Age'][df_train['Pclass']==3].describe()


# Distribuation of `Age` in each class showing than higher class has older passengers.

# In[ ]:


plt.hist([df_train['Age'][df_train['Survived']==0], df_train['Age'][df_train['Survived']==1]], 
         bins=15, stacked=True, color = ['#1f77b4', '#ff7f0e'])
plt.legend(['Survived 0', 'Survived 1'])
plt.title('The number of people by age divided into survival')
plt.grid()
plt.show()


# Most of newborns survived. They belong to the `Pclass` 2 and 3. In `Pclass` 1 there was a very few newborns.

# ## SibSp
# siblings/ spouses aboard the Titanic

# In[ ]:


df_train['SibSp'].unique()


# In[ ]:


df_train.groupby(['SibSp', 'Survived'])['Survived'].count().unstack(1).plot.bar(grid=True)
plt.yscale('log')
plt.title('The number of people by SibSp divided into Survived')
plt.show()


# People with smaller families survived more often.

# In[ ]:


df_train.groupby(['SibSp', 'Survived', 'Pclass'])['Survived'].count().unstack(1).plot.bar(grid=True)
plt.yscale('log')
plt.title('The number of people by SibSp divided into Survived and Pclass')
plt.show()


# Y scale is `log`!

# In[ ]:


df_train.groupby(['SibSp', 'Survived', 'Sex'])['Survived'].count().unstack(1).plot.bar(grid=True)
plt.yscale('log')
plt.title('The number of people by SibSp divided into Survived and Sex')
plt.show()


# Y scale is `log`!

# In general if people have larger family then chance for survive is lower.

# ## Parch
# parents / children aboard the Titanic

# In[ ]:


df_train['Parch'].unique()


# In[ ]:


df_train.groupby(['Parch', 'Survived'])['Survived'].count().unstack(1).plot.bar(grid=True)
plt.title('The number of people by Parch divided into Survived')
plt.yscale('log')
plt.show()


# Y scale is log!

# In[ ]:


df_train[['SibSp', 'Parch']].corr(method='spearman')
# pearson : standard correlation coefficient
# kendall : Kendall Tau correlation coefficient
# spearman : Spearman rank correlation


# Situaltion very simillar to the `SibSp` column.
# People with smaller families survived more often.

# ## Ticket

# In[ ]:


df_train['Ticket'].head()


# In[ ]:


df_train['Ticket'].apply(lambda x: x.split(' ')[0]).nunique()


# In[ ]:


df_train['Ticket'].apply(lambda x: len(x.split(' '))).unique()


# In[ ]:


df_train['Ticket_segment'] = df_train['Ticket'].apply(lambda x: len(x.split(' ')))


# In[ ]:


df_train.groupby(['Ticket_segment', 'Survived'])['Survived'].count().unstack(1).plot.bar(grid=True)
plt.title('The number of people by Ticket_segment divided into Survived')
plt.show()


# I will keep only this information (ticket's segments).

# ## Fare

# In[ ]:


df_train['Fare'].describe()


# In[ ]:


df_train['Fare'].plot.hist(grid=True)
plt.title('The number of people by Fare')
plt.yscale('log')
plt.show()


# Y scale is `log`!

# In[ ]:


plt.hist(
    [
        df_train['Fare'][df_train['Pclass']==1], 
        df_train['Fare'][df_train['Pclass']==2], 
        df_train['Fare'][df_train['Pclass']==3]
    ],
    bins=15, stacked=True)
plt.legend(['Pclass 1', 'Pclass 2', 'Pclass 3'])
plt.title('The number of people by Fare divided into Pclass')
plt.grid()
plt.show()


# In[ ]:


df_train['Fare'][df_train['Pclass']==1].describe()


# In[ ]:


df_train['Fare'][df_train['Pclass']==2].describe()


# In[ ]:


df_train['Fare'][df_train['Pclass']==3].describe()


# In[ ]:


plt.hist([df_train['Fare'][df_train['Survived']==0], df_train['Fare'][df_train['Survived']==1]], 
         bins=15, stacked=True, color = ['#1f77b4', '#ff7f0e'])
plt.legend(['Survived 0', 'Survived 1'])
plt.title('The number of people by Fare divided into survival')
plt.grid()
plt.show()


# Most of passangers from `Pclass` 3 belong to the first bin. This is the reason very large bin.

# ## Cabin

# In[ ]:


df_train['Cabin'].head()


# In[ ]:


df_train['Cabin'].isna().sum()


# There is a lot of empty values.

# In[ ]:


df_train['Cabin'][~df_train['Cabin'].isna()].apply(lambda x: x[0]).unique()


# Some information about titanic cabins (https://www.encyclopedia-titanica.org/cabins.html)

# In[ ]:


unique_cabin_char = df_train['Cabin'][~df_train['Cabin'].isna()].apply(lambda x: x[0]).unique().tolist()

for cabin_char in unique_cabin_char:
    df_train[cabin_char] = df_train['Cabin'].apply(lambda x: True if (isinstance(x, str) and x[0] == cabin_char) else False)


# In[ ]:


df_train[unique_cabin_char].sum()


# In[ ]:


df_train.groupby(['Survived']+unique_cabin_char)['Survived'].count().unstack(0)


# In[ ]:


ax = df_train.groupby(['Survived']+unique_cabin_char)['Survived'].count().unstack(0).plot.bar(grid=True)
ax.set_xticklabels((['NaN']+unique_cabin_char[::-1]))
plt.title('The number of people by Cabin first character divided into survival')
plt.legend(['Survived 0', 'Survived 1'])
plt.yscale('log')
plt.show()


# Y scale is `log`.
# 
# `T` is unique value. There is no evidance that more people lived there so I decided to remove this from a separate value.
# 
# Rest of cabines are very limited. I decided to connect `A` and `B`, `C` and `D`, `E` and `F` and `G`

# In[ ]:


update_unique_cabin_char = {'AB':['A', 'B'], 'CD': ['C', 'D'], 'EFG': ['E', 'F', 'G']}
df_train.drop(unique_cabin_char, axis='columns', inplace=True)

for keys, update_cabin_char in update_unique_cabin_char.items():
    df_train[f'Cabin_{keys}'] = df_train['Cabin'].apply(lambda x: True if (isinstance(x, str) and (x[0] in update_cabin_char)) else False)


# In[ ]:


update_unique_cabin_char = [f'Cabin_{x}' for x in list(update_unique_cabin_char.keys())]


# In[ ]:


ax = df_train.groupby(['Survived']+update_unique_cabin_char)['Survived'].count().unstack(0).plot.bar(grid=True)
ax.set_xticklabels((['NaN']+update_unique_cabin_char[::-1]))
plt.title('The number of people by Cabin first character divided into survival')
plt.yscale('log')
plt.show()


# Y scale is `log`!
# 
# The categories created are more numerous. 
# All categories have more people that surived (it could be missunderstood).

# ## Embarked

# In[ ]:


df_train['Embarked'].unique()


# In[ ]:


df_train['Embarked'].isna().sum()


# There are only 2 `NaN` values. 
# TODO: we need to deal with this (most popular valuefrom train set?)

# In[ ]:


df_train.groupby(['Embarked'])['Survived'].mean()


# In[ ]:


df_train.groupby(['Embarked', 'Survived'])['Survived'].count().unstack(1).plot.bar(grid=True)
plt.title('The number of people by Embarked divided into survival')
plt.show()


# In[ ]:


df_train.groupby(['Embarked', 'Survived', 'Pclass'])['Survived'].count().unstack(1).plot.bar(grid=True)
plt.title('The number of people by Embarked divided into survival and Pclass')
plt.yscale('log')
plt.show()


# Y scale is `log`!

# In[ ]:


df_train.groupby(['Embarked', 'Survived', 'Pclass', 'Sex'])['Survived'].count().unstack(1).plot.bar(grid=True)
plt.title('The number of people by Embarked divided into survival, Pclass and Sex')
# plt.yscale('log')
plt.show()


# The fewest people boarded in the Queenstown port.
# 
# Most people boarded in the Southampton port. Most of them are belong to the 3 class

# In[ ]:




