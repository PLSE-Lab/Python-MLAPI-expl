#!/usr/bin/env python
# coding: utf-8

# # Data Analysis

# ## Load data

# In[ ]:


import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head(10)


# ## Passenger class & Sex

# ### Pclass

# In[ ]:


train['Pclass'].value_counts()


# In[ ]:


print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())


# ### Sex

# In[ ]:


print (train[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean())


# In[ ]:


import seaborn as sns
sns.set(style="whitegrid")

# Draw a nested barplot to show survival for class and sex
g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")


# ## Age

# The age distribution for all passengers in the sample

# In[ ]:


train['Age'].hist(bins=60)


# The age distributions for people dead and survived

# In[ ]:


import matplotlib.pyplot as plt
f = sns.FacetGrid(train, col='Survived')
f.map(plt.hist, 'Age', bins=20)


# ## Family relations

# ### sibling or spouse

# In[ ]:


train['SibSp'].value_counts()


# In[ ]:


print (train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())


# In[ ]:


sns.catplot( x = 'SibSp', y = 'Survived',order=[0,1,2,3,4,5,6], height=4, kind = "point", data = train)


# ### Parents or Child

# In[ ]:


train['Parch'].value_counts()


# In[ ]:


print (train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())


# In[ ]:


sns.catplot( x = 'Parch', y = 'Survived',order=[0,1,2,3,4,5,6], height=4, kind = "point", data = train)


# ## Fare 

# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# ## Embarked

# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())


# In[ ]:


sns.set(style="whitegrid")
h = sns.catplot(x="Embarked", y="Survived", data=train,
                height=4, kind="bar", palette="muted")
h.despine(left=True)
h.set_ylabels("survival probability")


# # Data cleaning

# Drop variables that will not be used 

# In[ ]:


train = train.drop(['Name', 'Ticket', 'Cabin'],axis=1)
test = test.drop(['Name', 'Ticket','Cabin'],axis=1)
train.head(10)


# Convert categorical data into numerical data 

# In[ ]:


for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})


# In[ ]:


train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')


# In[ ]:


for df in [train,test]:
    df['Embarked_spots']=df['Embarked'].map({'S':0,'C':1, 'Q':2})


# Fill missing values in Age column with random numbers

# In[ ]:


import numpy as np

average_age_train = train["Age"].mean()
std_age_train = train["Age"].std()
count_nan_age_train = train["Age"].isnull().sum()

average_age_test = test["Age"].mean()
std_age_test = test["Age"].std()
count_nan_age_test = test["Age"].isnull().sum()

rand_1 = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train, size = count_nan_age_train)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

train["Age"][np.isnan(train["Age"])] = rand_1
test["Age"][np.isnan(test["Age"])] = rand_2

train['Age'] = train['Age'].astype(int)
test['Age'] = test['Age'].astype(int)


# Fill missing values in Fare column with the median value

# In[ ]:


test["Fare"].fillna(test["Fare"].median(), inplace=True)


# In[ ]:


train.head(10)


# In[ ]:


test.head(10)


# # Modeling and Prediction

# Select feature column names and target variable we are going to use for training

# In[ ]:


features = ['Pclass','Age','Sex_binary','SibSp','Parch','Fare', 'Embarked_spots']
target = 'Survived'


# In[ ]:


train[features].head(5)


# In[ ]:


train[target].head(3).values


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

clf= DecisionTreeClassifier(max_depth=3,min_samples_leaf=2) 
clf.fit(train[features], train[target])

predictions = clf.predict(test[features])


# In[ ]:


predictions


# Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

# In[ ]:


submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})
submission.head()


# Convert DataFrame to a csv file that can be uploaded

# In[ ]:


filename = 'Titanic Predictions 8.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

