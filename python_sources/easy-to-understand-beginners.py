#!/usr/bin/env python
# coding: utf-8

# ## Beginners attempt to Titanic problem

# Importing all packages required for data preprocessing and loading datasets:

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
# Input data files are available in the read-only "../input/" directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Checking a few samples in the data to get an idea:

# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
print(train.head(2))


# Checking details about each attributes:

# In[ ]:


print(train.describe())


# Checking if there are any null/void entries in any cell, and if so then how many:

# In[ ]:


print(pd.isnull(train).sum())
comb=[train,test]


# ### Feature analysis:

# ##### Feature 'Sex':

# In[ ]:


sns.barplot(x="Sex",y="Survived", data=train)

print(train[['Survived','Sex']].groupby(['Sex']).mean())
pd.crosstab(train['Sex'],train['Survived'])


# We can see from the barplot above that a higher percentage of female have survived compared to male

# #####  Feature 'Pclass':

# In[ ]:


sns.barplot(x="Pclass",y="Survived",data=train)

print(train["Survived"][train["Pclass"]==1].value_counts())
print(train["Survived"][train["Pclass"]==2].value_counts())
print(train["Survived"][train["Pclass"]==3].value_counts())

#train[['Survived','Pclass']].sum()


# We can see from the barplot above that a better chances of people surviving if they are from better PClass

# ##### Feature 'Parch' and 'SibSp':

# In[ ]:


# I think it would be more sensible to make a feature is a person is alone or with family.
# So to make that I combine parent children and Sibling spouse features.
# If after combining their values it is 0 then the person is travelling by himself/herself.

train['FamilySize']=train['Parch']+train['SibSp']
test['FamilySize']=test['Parch']+test['SibSp']

train['Alone']=0
test['Alone']=0
train.loc[train['FamilySize']==0,'Alone']=1
test.loc[test['FamilySize']==0,'Alone']=1
sns.barplot(x='Alone',y='Survived',data=train)


# People alone are more likely to die compared to people with family

# ##### Feature 'Age':

# In[ ]:


# In Age we have seen that there are huge number of values that are missing so we need to find a way to fill those
# As age can be one of the important features

#So first we categorise the KNOWN age into few bins with the missing(filled with -0.5) ones into the category 'UNKNOWN'

train['Age']=train['Age'].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)
#train['AgeGroup'].unique()
#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)


# ##### Feature 'Title':
# 
# Here we shall extract titles from 'Name' and use them for further use as the Name is not of much use

# In[ ]:


train['title']=train['Name'].str.extract(pat = '([A-Za-z]+)\.') 
test['title']=test['Name'].str.extract(pat = '([A-Za-z]+)\.') 
print(train['title'].value_counts())
#pd.crosstab(train['title'],train['Survived'])


# In[ ]:


#Trying to reduce the categories into more logical terms

for data in comb:
    data['title']=data['title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
    data['title']=data['title'].replace(['Mme'],'Mrs')
    data['title']=data['title'].replace(['Mlle','Ms'],'Miss')
    data['title']=data['title'].replace(['Countess','Sir','Lady'],'Royal')

train['title'].value_counts()
pd.crosstab(train['title'],train['Survived'])


# For the missing Age I have thought of replacing them with the mode of age-group occuring for the that corresponding title

# In[ ]:


title_map={'Mr':1, 'Master':2, 'Miss':3, 'Mrs':4, 'Rare':5, 'Royal':6}
for dataset in comb:
    dataset['title'] = dataset['title'].map(title_map)

print(train[['title']].head())
#Calculating which age group occurs max for a particular title

mr_age = train[train["title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train[train["title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train[train["title"] == 3]["AgeGroup"].mode() #Adult
master_age = train[train["title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train[train["title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train[train["title"] == 6]["AgeGroup"].mode() #Senior


# In[ ]:


age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["title"][x]]

for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["title"][x]]


# ##### Feature 'Cabin':
# 
# I am making the feature 'Cabin' to check if a person has a cabin or not. Its number is irrelevant in my case.

# In[ ]:


train['isCabin']=0
train.loc[train['Cabin'].notnull(),'isCabin']=1
test['isCabin']=0
test.loc[test['Cabin'].notnull(),'isCabin']=1

print(train[['isCabin','Survived']].groupby(['isCabin']).mean())
sns.barplot(x='isCabin',y='Survived',data=train)


# We can see that people with Cabin are more likely to survive than the ones without

# In[ ]:


print(train[['Embarked','Survived']].groupby(['Embarked']).sum())
print(train[['Embarked','Survived']].groupby(['Embarked']).mean())
sns.barplot(x='Embarked',y='Survived',data=train)


# In[ ]:


# Since a very high percentage of passengers have 'Embarked' as S we 
# are filling the missing 2 as 'S' and also mapping them

train['Embarked']=train['Embarked'].fillna('S')
train['Embarked']=train['Embarked'].map({"S":1, "C":2, "Q":3})
test['Embarked']=test['Embarked'].map({"S":1, "C":2, "Q":3})


# In[ ]:


age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup']=train['AgeGroup'].map(age_mapping)
test['AgeGroup']=test['AgeGroup'].map(age_mapping)


# In[ ]:


train['Sex']=train['Sex'].map({"male":0, "female":1})
test['Sex']=test['Sex'].map({"male":0, "female":1})


# ##### Feature 'Fare':
# 
# The missing fares are filled with the mean of that corresponding 'Pclass'

# In[ ]:


for x in range(len(test['Fare'])):
    if pd.isnull(test['Fare'][x]):
        test['Fare'][x] = round(train['Fare'][train['Pclass']==test['Pclass'][x]].mean(), 4)


# FareBand will categorise the fare into 4 categories

# In[ ]:


train['FareBand']=pd.qcut(train['Fare'], 4, labels=[1,2,3,4])
test['FareBand']=pd.qcut(test['Fare'], 4, labels=[1,2,3,4])
sns.barplot(x='FareBand', y='Survived', data=train)


# Now that we have seen the features, we shall move to the training. For that we first drop the columns that we shall not use.

# In[ ]:


train1=train.drop(columns=['Cabin','Ticket','Age','FamilySize','SibSp','Parch','Name','Fare','PassengerId','Survived','AgeGroup'])
test1=test.drop(columns=['Cabin','Ticket','Age','FamilySize','SibSp','Parch','Name','Fare','PassengerId','AgeGroup'])


# In[ ]:


train1.head()


# In[ ]:


test1.head()


# In[ ]:


from sklearn.model_selection import train_test_split

target = train['Survived']
x_train,x_val,y_train,y_val = train_test_split(train1,target,test_size = 0.2, random_state=0)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(x_train,y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100,2)
print(acc_gaussian)


# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val)*100, 2)
print(acc_logreg)


# In[ ]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[ ]:


from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[ ]:


from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[ ]:


from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


sns.barplot(x='Score', y='Model', data=models.sort_values(by=["Score"]), color="y")

Creating the submission file using the best model:
# In[ ]:


ids = test['PassengerId']
predictions = gbk.predict(test1)

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submissions.csv', index=False)

