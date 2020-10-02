#!/usr/bin/env python
# coding: utf-8

# **#Titanic Survival Predictions (Beginner)**
# 
# 1) ImportingLibraries
# 2) Read In and Explore the Data
# 3) Data Analysis
# 4) Data Visualization
# 5) Cleaning Data
# 6) Model Building
# 7)Creating Submission File
# 
# **Description of the Data Fields:**
# survival - Survival (0 = No; 1 = Yes)
# class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# name - Name
# sex - Sex
# age - Age
# sibsp - Number of Siblings/Spouses Aboard
# parch - Number of Parents/Children Aboard
# ticket - Ticket Number
# fare - Passenger Fare
# cabin - Cabin
# embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# **1)Importing the Libraries**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 


# **2) Read In and Explore the Data**

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


#To view the Train data
train


# In[ ]:


#To view the test data
test


# In[ ]:


#To get the no.of columns and rows in train data
train.shape


# In[ ]:


#To get the no.of columns and rows in test data
test.shape


# ** 3)Data Analysis**

# In[ ]:


train.columns


# In[ ]:


# To get the top 5 column values
train.head()


# In[ ]:


train.describe()


# In[ ]:


#To get the missing values
print(pd.isnull(train).sum())


# 4) Data Visualization

# In[ ]:


sns.countplot(x='Survived',data=train)


# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=train)


# By the above plot we can get to know that females have a much higher chance of survival than males. The Sex feature is essential in our predictions.

# In[ ]:


#grouping the sex and survived to get the count of the survived and not survived by sex
train.groupby(['Survived','Sex'])['Survived'].count()


# In[ ]:


# To get the percentage of the survived by sex

print("% of women survived: " , train[train.Sex == 'female'].Survived.sum()/train[train.Sex == 'female'].Survived.count())
print("% of men survived:   " , train[train.Sex == 'male'].Survived.sum()/train[train.Sex == 'male'].Survived.count())


# In[ ]:


#Grouping Pclass and survived
train.groupby(['Survived','Pclass'])['Survived'].count()


# In[ ]:


#To get the percentage of the survival rate wrt Pclass
print("% of Pclass 1 survived",train[train.Pclass==1].Survived.sum()/train[train.Pclass==1].Survived.count())
print("% of Pclass 2 survived",train[train.Pclass==2].Survived.sum()/train[train.Pclass==2].Survived.count())
print("% of Pclass 3 survived",train[train.Pclass==3].Survived.sum()/train[train.Pclass==3].Survived.count())


# By the above output we can get to know that Pclass 1  have a much higher chance of survival than Pclass 2 and Pclass 3. The Pclass feature is essential in our predictions.

# In[ ]:


#grouping SibSp and Survived
train.groupby(['Survived','SibSp'])['Survived'].count()


# In[ ]:


print("% of sibsp 0 survived",train[train.SibSp==0].Survived.sum()/train[train.SibSp==0].Survived.count())
print("% of sibsp 1 survived",train[train.SibSp==1].Survived.sum()/train[train.SibSp==1].Survived.count())
print("% of sibsp 2 survived",train[train.SibSp==2].Survived.sum()/train[train.SibSp==2].Survived.count())


# By the above output it is clear that people with no siblings or spouses were less  likely to survive than those with one or two. 

# In[ ]:


train.groupby(['Survived','Parch'])['Survived'].count()


# In[ ]:


print("% of Parch 0 survived",train[train.Parch==0].Survived.sum()/train[train.Parch==0].Survived.count())
print("% of Parch 1 survived",train[train.Parch==1].Survived.sum()/train[train.Parch==1].Survived.count())
print("% of Parch 2 survived",train[train.Parch==2].Survived.sum()/train[train.Parch==2].Survived.count())


# By the above output people traveling alone are less likely to survive than those with 1-2 parents or children.

# In[ ]:


train.groupby(['Survived','Age'])['Survived'].count()


# In[ ]:


#sorting the ages into logical categories
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()


# Babies are more likely to survive than any other age group.

# In[ ]:


train.groupby(['Survived','Embarked'])['Survived'].count()


# In[ ]:


print("% of Embarked c survived",train[train.Embarked=='C'].Survived.sum()/train[train.Embarked=='C'].Survived.count())
print("% of Embarked Q survived",train[train.Embarked=='Q'].Survived.sum()/train[train.Embarked=='Q'].Survived.count())
print("% of Embarked S survived",train[train.Embarked=='S'].Survived.sum()/train[train.Embarked=='S'].Survived.count())


# People who started from the Port of Embarkation C = Cherbourg are more likely to survival than others

# In[ ]:


train.groupby(['Survived','Fare'])['Survived'].count()


# In[ ]:


train.groupby(['Survived','Cabin'])['Survived'].count()


# **5)Data Cleaning**

# In[ ]:


print(pd.isnull(train).sum())


# In[ ]:


test.describe()


# In[ ]:


test.columns


# In[ ]:


train.columns


# In[ ]:


train.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1,inplace=True)


# In[ ]:


id=test['PassengerId']
test.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1,inplace=True)


# In[ ]:


train = train.fillna({"Embarked": "S"})


# In[ ]:


print(pd.isnull(train).sum())


# In[ ]:


print(pd.isnull(test).sum())


# In[ ]:


train


# **6)Model Building**

# In[ ]:


import statsmodels.formula.api as sm


# In[ ]:


logit_model = sm.logit('Survived~Pclass+Sex+SibSp+Parch+Embarked+AgeGroup',data = train).fit()


# In[ ]:


logit_model.summary()


# In[ ]:


logit_model.params


# In[ ]:


predictions_test=np.round(logit_model.predict(test))


# In[ ]:


predictions_test


# In[ ]:


predict_train=np.round(logit_model.predict(train))


# In[ ]:


predict_train


# In[ ]:


from sklearn.metrics import accuracy_score 
Accuracy_Score = accuracy_score(train['Survived'],predict_train)


# In[ ]:


Accuracy_Score


# In[ ]:


output = pd.DataFrame({ 'PassengerId' : id, 'Survived': predictions_test})


# In[ ]:


output.to_csv('submission.csv', index=False)


# In[ ]:


output


# In[ ]:




