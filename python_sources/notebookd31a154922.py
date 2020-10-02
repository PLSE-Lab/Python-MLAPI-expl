#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df    = pd.read_csv("../input/test.csv")

pId_df = test_df['PassengerId']
titanic_df = train_df.drop(['PassengerId','Name','Ticket', 'Embarked', 'Cabin'], axis=1)
test_df = test_df.drop(['PassengerId','Name','Ticket', 'Embarked', 'Cabin'], axis=1)


# dummies = pd.get_dummies(titanic_df['Sex'])


# In[ ]:


from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
#to convert into numbers
titanic_df.Sex = le_sex.fit_transform(titanic_df.Sex)
test_df.Sex = le_sex.fit_transform(test_df.Sex)


# In[ ]:


test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)


# get average, std, and number of NaN values in titanic_df
average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# fill NaN values in Age column with random values generated
titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

# convert from float to int
titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age']    = test_df['Age'].astype(int)


# In[ ]:


titanic_df.info()
print("----------------------------")
test_df.info()


# In[ ]:


#define training and testing sets

X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)


# In[ ]:


#Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svc.score(X_train, Y_train)


# In[ ]:


# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
knn.score(X_train, Y_train)


# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

gaussian.score(X_train, Y_train)


# In[ ]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
clf.score(X_train, Y_train)


# In[ ]:



submission = pd.DataFrame({
        "PassengerId": pId_df,
        "Survived": Y_pred
    })
 
submission.to_csv('titanic.csv', index=False)
print(submission)


# In[ ]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(algorithm='adam', alpha=1e-5, hidden_layer_sizes=(10, 10, 10), random_state=1, activation='tanh')
clf = clf.fit(X_train, Y_train)
clf.score(X_train, Y_train)


# In[ ]:


# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
coeff_df

