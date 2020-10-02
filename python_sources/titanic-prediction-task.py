#!/usr/bin/env python
# coding: utf-8

# In[32]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


#ignore warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
id = test["PassengerId"]

#explorative data analysis to get some insight
train.info()


# In[33]:


train.head(10)


# In[34]:


sns.catplot(x="Survived", hue="Sex", kind="count", data=train)
plt.show()


# In[35]:


sns.catplot(x="Survived", hue="Pclass", kind="count", data=train)
plt.show()


# In[36]:


g1 = train.loc[train["Survived"] == 1 , "Age"]
g1 = g1[g1.notna()]
g2 = train.loc[train["Survived"] == 0 , "Age"]
g2 = g2[g2.notna()]

sns.distplot(g1, color="red")
sns.distplot(g2, color="blue")
plt.show()


# In[37]:


g1 = train.loc[(train["Survived"] == 1) & (train["Fare"] < 100) , "Fare"]
g1 = g1[g1.notna()]
g2 = train.loc[(train["Survived"] == 0) & (train["Fare"] < 100), "Fare"]
g2 = g2[g2.notna()]

sns.distplot(g1, color="red")
sns.distplot(g2, color="blue")
plt.show()


# In[38]:


sns.catplot(x="Survived", hue="Embarked", kind="count", data=train)
plt.show()


# In[39]:


sns.catplot(x="Survived", hue="SibSp", kind="count", data=train)
plt.show()


# In[40]:


sns.catplot(x="Survived", hue="Parch", kind="count", data=train)
plt.show()


# In[41]:


#drop and combine train and test
df = pd.concat([train.drop(["PassengerId", "Survived"], axis=1), test.drop("PassengerId", axis=1)], axis=0)
df.info()


# In[42]:


df.head()


# In[43]:


#fill in missing values
#Embarked: use the value with the highest frequency 
print(df["Embarked"].value_counts())
df.fillna({"Embarked" : "S"}, inplace=True)

#Fare: use median because a lot of outliers
df.boxplot(column=["Fare"])
plt.show()
df.fillna({"Fare" : df["Fare"].median()}, inplace=True)

#Family size: constructed as new feature SibSp + Parch
df["FamilySize"] = df["SibSp"] + df["Parch"]
df = df.drop(["SibSp", "Parch"], axis=1)

#further drop Name, Ticket, Cabin
df = df.drop(["Ticket"], axis=1)
df.info()


# In[44]:


#reconstruct name feature
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=True)

titles = {'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Don':4, 'Rev':4, 'Dr':4, 'Mme':1, 'Ms':0,
            'Major':4, 'Lady':5, 'Sir':4, 'Mlle':2, 'Col':4, 'Capt':4, 'Countess':5, 'Jonkheer':4,'Dona':5}


df['Title'] = df['Title'].map(titles)
df = df.drop("Name", axis=1)

#sns.countplot(y="Title", data=df)
df_ = df[:len(train)]
df_["Survived"] = train["Survived"]
sns.catplot(x="Survived", hue="Title", kind="count", data=df_)
plt.show()
df.head()


# In[45]:


#cabine
#Cabin: drop because a lot of values are missing
df = df.drop("Cabin", axis=1)
df.head()


# In[46]:


#encode features
categorical_vars = ["Sex", "Pclass", "Embarked", "Title"]
df = pd.get_dummies(df, columns=categorical_vars)
df.head(10)


# In[47]:


#use boxplot to find outliers
df.boxplot(column=["Age"])
plt.show()

#use linear regression for predicting missing age-values
from sklearn.linear_model import LinearRegression

#remove null values and outliers for model building
X = df.dropna(how="any")
X = X[X["Age"] < 64]
y = X["Age"]
X = X.drop("Age", axis=1)
reg = LinearRegression().fit(X, y)

#predict age values where no values are available
X = df
X = X[df["Age"].isnull()]
X = X.drop("Age", axis=1)

pred = np.round(reg.predict(X))
df.loc[df["Age"].isnull(), "Age"] = pred


# In[48]:


#normalization of features age and fare
df["Age"] = df["Age"].apply(lambda x: (x - df["Age"].mean()) / df["Age"].std()) 
df["Fare"] = df["Fare"].apply(lambda x: (x - df["Fare"].mean()) / df["Fare"].std()) 

#split into train and test again
y = train["Survived"]
train = df[:len(train)]
test = df[len(train):]
df.head(10)


# In[49]:


#classification task
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#cross validation using 500 trees and different number of features
params = {'n_estimators': [100], 'max_features': [3, 5, 7, 9, 11, 13, 15, 17]}
rf = RandomForestClassifier(random_state=0) 
clf = GridSearchCV(rf, params, cv=10)
clf.fit(train, y)

#print best parameter
print(clf.best_params_)

#plot training accuracy
cv_acc = clf.cv_results_['mean_test_score']
plt.plot([3, 5, 7, 9, 11, 13, 15, 17], cv_acc, "-o")
plt.ylabel('accuracy')
plt.xlabel('nr. of features')
plt.show()

#apply best settings on test set
pred = clf.predict(test)


# In[50]:


#performing logistic regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='lbfgs', max_iter = 10000)
params = {'C': [0.01, 0.1, 0.5, 1, 3, 4, 5, 6, 7, 10]}
clf = GridSearchCV(lr, params, cv=10)
clf.fit(train, y)

#print best parameter
print(clf.best_params_)

#plot training accuracy
cv_acc = clf.cv_results_['mean_test_score']
plt.plot([0.01, 0.1, 0.5, 1, 3, 4, 5, 6, 7, 10], cv_acc, "-o")
plt.ylabel('accuracy')
plt.xlabel('C')
plt.show()

#apply best settings on test set
pred = clf.predict(test)


# In[51]:


from sklearn.svm import SVC  
svm = SVC(kernel="poly", gamma="auto")
params = {'C': [0.1, 0.5, 1., 1.5, 2., 3., 5., 10.], 'degree': [2]}

clf = GridSearchCV(svm, params, cv=10)
clf.fit(train, y)

#print best parameter
print(clf.best_params_)

#plot training accuracy
cv_acc = clf.cv_results_['mean_test_score']

plt.plot([0.1, 0.5, 1., 1.5, 2., 3., 5., 10.], cv_acc, "-o")
plt.ylabel('accuracy')
plt.xlabel('C')
plt.show()


# In[52]:


from sklearn.svm import SVC  
svm = SVC(kernel="rbf", gamma="auto")
params = {'C': [0.1, 0.2, 0.3, 0.5, 1., 1.5, 2., 3., 5., 10.]}

clf = GridSearchCV(svm, params, cv=10)
clf.fit(train, y)

#print best parameter
print(clf.best_params_)

#plot training accuracy
cv_acc = clf.cv_results_['mean_test_score']

plt.plot([0.1, 0.2, 0.3, 0.5, 1., 1.5, 2., 3., 5., 10.], cv_acc, "-o")
plt.ylabel('accuracy')
plt.xlabel('C')
plt.show()


# In[53]:


#apply best setting from svm on test set
pred = clf.predict(test)
pred = [int(x) for x in pred]

out = pd.DataFrame({'PassengerId':id, 'Survived':pred})
#print(out)
out.to_csv("submission.csv", index=False)

