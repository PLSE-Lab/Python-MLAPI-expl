#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **Preparation of a dataframe for the train and test data:**

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
#train_df.head()

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
#test_df.head()

traintest_df = [train_df, test_df]


# **Which features are obviously available in the dataset "train_df?**

# In[ ]:


print(train_df.columns.values)


# **What is the percentage of women who have survived?**

# In[ ]:


women = train_df.loc[train_df.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("Percent of women who survived:", rate_women)


# **What is the percentage of men who have survived?**

# In[ ]:


men = train_df.loc[train_df.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("Percent of men who survived:", rate_men)


# In[ ]:


train_df.describe(include=['O'])


# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# **Checks based on the problem description and the first data checks:**
# 
# 1.	Women (Sex=female) survived with 75% rate (0.7420382165605095)
# 3.	1st class passengers (Pclass=1) survived with 63% rate (0.629630)

# **Play around with the names and titels:**

# In[ ]:


for dataset in traintest_df:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])


# **Group name and titels:**

# In[ ]:


for dataset in traintest_df:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Countess','Lady','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# **Checks II based on the problem description and the first data checks:**
# 
# 1. Women (Sex=female) survived with 75% rate (0.7420382165605095)
# 2. 1st class passengers (Pclass=1) survived with 63% rate (0.629630)
# 3. If Title == Master survival with 58% rate (0.575000)
# 

# In[ ]:


title_clear = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in traintest_df:
    dataset['Title'] = dataset['Title'].map(title_clear)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()


# **Actions: **
# 1. drop the Name feature from training and testing datasets
# 2. drop PassengerId feature in the training dataset

# In[ ]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
traintest_df = [train_df, test_df]
train_df.shape, test_df.shape


# **Action: **
# 
# 1. converting Sex feature to a new feature called Sex where female=1 and male=0.

# In[ ]:


for dataset in traintest_df:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# **Action: **
# 1. drop feature "Age" 
# 2. drop feature "SibSp"
# 3. drop feature "Parch"
# 4. drop feature "Ticket" 
# 5. drop feature "Fare" 
# 6. drop feature "Cabin" 
# 7. drop feature "Embarked"

# In[ ]:


train_df = train_df.drop(['Age', 'SibSp', 'Parch','Ticket', 'Fare','Cabin','Embarked'], axis=1)
traintest_df = [train_df, test_df]
train_df.head()


# In[ ]:


test_df = test_df.drop(['Age', 'SibSp', 'Parch','Ticket', 'Fare','Cabin','Embarked'], axis=1)
traintest_df = [train_df, test_df]
test_df.head()


# **Prediction model with Random Forest Classifier: **

# In[ ]:


df1_train = train_df.drop("Survived", axis=1)
df2_train = train_df["Survived"]
df1_test  = test_df.drop("PassengerId", axis=1).copy()
df1_train.shape, df2_train.shape, df1_test.shape


# In[ ]:


random_forest_1 = RandomForestClassifier(n_estimators=100)
random_forest_1.fit(df1_train, df2_train)
df2_pred = random_forest_1.predict(df1_test)
random_forest_1.score(df1_train, df2_train)
rate_random_forest = round(random_forest_1.score(df1_train, df2_train) * 100, 2)
rate_random_forest


# In[ ]:


my_submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],"Survived": df2_pred})
my_submission.to_csv('my_submission_AG5.csv', index=False)
print('Submission was successfully!')


# In[ ]:




