#!/usr/bin/env python
# coding: utf-8

# Titanic Survival prediction
# 
# Hello, this is my first kernel here, I am beginning my studies of data science, more focused on machine learning.
#         
# Please feel free to comment.

# **Importing libraries.
# **
# 
# First, we will use the block below to load some Python libraries who will be used at this kernel.
# 

# In[ ]:


#data analysis libraries

import numpy as np
import pandas as pd

#visualization libraries

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# **Reading the data.**

# In[ ]:


#train data
url = '../input/titanic/train.csv'
train = pd.read_csv(url)

url = '../input/titanic/test.csv'
test = pd.read_csv(url)


# **Showing the training data.**

# In[ ]:


train.describe(include="all")


# **Data analysis.**

# In[ ]:


#show the first ten rows of the dataset
train.sample(10)


# In[ ]:


print(pd.isnull(train).sum())


# analysing the dataset, we find four numerical features(Age(continuous), fare(continous), SibSp(discrete), Parch(discrete)), four categorical features(Survived, Sex, Embarked, Pclass) and two alphanumeric features(Ticket and Cabin).
# 
# There are a total of 891 passengers on training data; The column 'Age' has 714 values, and consequently 177 NaN values. I think these values is very important for our model and we should try to fullfill it; Only 204 rows has data on 'Cabin' column, since more than 70% of the column doesn't have data, we should ignore it; Only 2 values is missing on Embarked feature, probably won't be a problem.
# 
# 

# We will drop Ticket feature, It may not have correlation with Survival.
# 
# Cabin feature contains many null values and we will drop it.
# 
# Name and Passengerid may not contribute to survival and will be dropped.

# **Data visualization
# **

# In[ ]:


#barplot of survivals by sex
sns.barplot(x="Sex", y="Survived", data=train)


# Barplot show us that females have more chance to survive.

# In[ ]:


#Barplot of Pclass.
sns.barplot(x="Pclass", y="Survived", data=train)


# Passengers of higher classes have better chance of survival.

# In[ ]:


train.drop(['PassengerId'], 1).hist(figsize=(25,18))
plt.show()


# Only about 37% of passengers survived.
# 
# Majority of fare tickets are below 50.
# 
# The people have better chance to survive if they are alone (sibsp and parch).

# **Cleaning data**

# Filling the Embarked feature missing values.

# In[ ]:


print("Number of people embarking in Southampton (S):")
southampton = train[train["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = train[train["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = train[train["Embarked"] == "Q"].shape[0]
print(queenstown)


# In[ ]:


#Filling the missing values with S (where the majority embarked).
train = train.fillna({"Embarked": "S"})


# In[ ]:


test = test.fillna({"Embarked": "S"})


# In[ ]:


#Filling the missing values based on mean fare for that Pclass
#PS. Only the test database has null values
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
    


# In[ ]:


#converting to numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# In[ ]:


#Drop Ticket and Cabin
train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)


# There is high percentage of values missing in the age feature, so it makes no sense to fill these gaps with the same value as we did before.
# 
# We will replace missing values by the mean age of passengers who belong to the same group of class, sex and family.

# In[ ]:


#And fill the missing Age values
train['Age'] = train.groupby(['Pclass','Sex','Parch','SibSp'])['Age'].transform(lambda x: x.fillna(x.mean()))
train['Age'] = train.groupby(['Pclass','Sex','Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))
train['Age'] = train.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


test['Age'] = test.groupby(['Pclass','Sex','Parch','SibSp'])['Age'].transform(lambda x: x.fillna(x.mean()))
test['Age'] = test.groupby(['Pclass','Sex','Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))
test['Age'] = test.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


#count the missing values of the Age feature
print(pd.isnull(train['Age']).sum())


# In[ ]:


print(pd.isnull(test['Age']).sum())


# In[ ]:


#Here we will create a Title column
train['Title'] = pd.Series((name.split('.')[0].split(',')[1].strip() for name in train['Name']), index=train.index)
train['Title'] = train['Title'].replace(['Lady','the Countess','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
train['Title'] = train['Title'].replace(['Mlle','Ms'], 'Miss')
train['Title'] = train['Title'].replace('Mme','Mrs')
train['Title'] = train['Title'].map({"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5})

test['Title'] = pd.Series((name.split('.')[0].split(',')[1].strip() for name in test['Name']), index=test.index)
test['Title'] = test['Title'].replace(['Lady','the Countess','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
test['Title'] = test['Title'].replace(['Mlle','Ms'], 'Miss')
test['Title'] = test['Title'].replace('Mme','Mrs')
test['Title'] = test['Title'].map({"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5})


# In[ ]:


#Drop name and PassengerId
train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)


# Now We will convert sex feature to numerical.

# In[ ]:


#converting sex feature to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# Converting the embarked feature:

# In[ ]:


#same to Embarked feature
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)


# Now that our data is numeric, We can check the correlation between the features.

# In[ ]:


plt.subplots(figsize = (12, 12))
sns.heatmap(train.corr(), annot = True, linewidths = .5)


# Verifying the data before models creation.

# In[ ]:


train.head()


# **Training models**

# In[ ]:


#splitting training data to test accuracy.
from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived'],axis = 1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.25, random_state = 0)


# We will test these models:
# 
#     Decision Tree Classifier
#     Gaussian Naive Bayes
#     Gradient Boosting Classifier
#     KNN
#     Logistic Regression
#     Perceptron
#     Random Forest Classifier
#     Stochastic Gradient Descent
#     Support Vector Machines
# 

# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[ ]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[ ]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[ ]:


#KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[ ]:


#Logistic Reggression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[ ]:


#Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)


# In[ ]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[ ]:


#Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[ ]:


#Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[ ]:


#Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[ ]:




models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 'SVC',  
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron, acc_linear_svc, acc_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# Decision Tree has the higher score.

# **Submission file**

# In[ ]:


ids = test['PassengerId']
predictions = gbk.predict(test.drop('PassengerId', axis = 1))
submission = pd.DataFrame({'PassengerId' : ids, 'Survived' : predictions})
submission.to_csv('submission.csv', index = False)

