#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


# # Load the data

# In[96]:


train_df = pd.read_csv('../input/train.csv') # training portion
test_df = pd.read_csv('../input/test.csv')
combine_df = [train_df, test_df] # all data 
train_df.head()


# Don't know yet what are SibSp, Parch, Fare and Embarked.

# # Brief look at the data

# In[97]:


print(train_df.columns.values) #get the list of all column names


# The names are ok, let's proceed with the variable types.
# 
# Numerical: Age, Fare (cont), SibSp, Parch (discrete).
# 
# Categorical: Survived, Sex, Embarked (no order) + Pclass (ordinal) .
# 
# Mixed: Cabin, Ticket.
# 
# 

# In[98]:


train_df.info()
print('_'*40)
test_df.info()


# So, there are massive missings of Age and Cabin, and a few of Embarked in the training sample; and, in the test sample, there are also many missings of Age and Cabin, and 1 missing of  Fare.

# # Descriptive statistics

# In[99]:


train_df.describe()


# In[100]:


train_df.describe(include='O')


# In[101]:


fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(train_df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()


# Seemingly, the most important determinant of survival are: Pclass and Fare.
# 
# # Data preparation
# ### Delete what doesn't seem useful
# Since I don't really understand the value of Cabin and Ticket variables (and, in addition, Cabin has lots of missings), I'll start with those variables dropped out of samples.

# In[102]:


print("Before", train_df.shape, test_df.shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


# ### Add a new feature derived from the Name 
# These is also usefull info in the variable Name that can be exploited.

# In[103]:


for obs in combine:
    obs['Title'] = obs.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])


# Some title are frequent, while other are rare. Let's combine all of the rare categories into one. Also, let's fix all typos which are visible, so we will have a smaller number of reasonable categories.

# In[104]:


for obs in combine:
    obs['Title'] = obs['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    obs['Title'] = obs['Title'].replace('Mlle', 'Miss')
    obs['Title'] = obs['Title'].replace('Ms', 'Miss')
    obs['Title'] = obs['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# Then  I recode Title to be an ordinal variable: the higher is the values, the higher probability of survival:

# In[105]:


title_mapping = {"Mr": 1, "Rare": 2, "Master": 3, "Miss": 4, "Mrs": 5}
for obs in combine:
    obs['Title'] = obs['Title'].map(title_mapping)
    obs['Title'] = obs['Title'].fillna(0)

train_df.head()


# Ok, now we can drop Name...and PassengerId also.

# In[106]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# ### Convert Sex from string to a dummy variable:

# In[107]:


for obs in combine:
    obs['Sex'] = obs['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# ### Imputation of Age:

# In[109]:


my_imputer = SimpleImputer()
train_df['Age'] = pd.DataFrame(my_imputer.fit_transform(train_df['Age'].values.reshape(-1,1)))
test_df['Age'] = pd.DataFrame(my_imputer.fit_transform(test_df['Age'].values.reshape(-1,1)))


# ### Simple Imputation of Embarked:
# here we use 'most frequent' strategy, for string variable Embarked

# In[110]:


my_imputer_cat = SimpleImputer(strategy = 'most_frequent')
train_df['Embarked'] = pd.DataFrame(my_imputer_cat.fit_transform(train_df['Embarked'].values.reshape(-1,1)))
test_df['Embarked'] = pd.DataFrame(my_imputer_cat.fit_transform(test_df['Embarked'].values.reshape(-1,1)))


# ### Simple Imputation of Fare (test sample only)

# In[111]:


test_df['Fare'] = pd.DataFrame(my_imputer.fit_transform(test_df['Fare'].values.reshape(-1,1)))


# In[112]:


combine = [train_df, test_df]


# ### Converting Embarked to numeric:

# In[113]:


pd.crosstab(train_df['Survived'], train_df['Embarked'])


# In[114]:


for obs in combine:
    obs['Embarked'] = obs['Embarked'].map( {'S': 0, 'Q': 1, 'C': 2} ).astype(int)


# In[115]:


train_df.head()


# # Building Models

# In[116]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# ### Support Vector Machines

# In[ ]:


# Support Vector Machines
from sklearn.svm import SVC, LinearSVC

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# ### Gaussian Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# ### Perceptron

# In[ ]:


from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# ### Linear SVC

# In[ ]:


from sklearn.svm import  LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# ### Decision Trees

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# ### Stochastic Gradient Descent Classification

# In[137]:


from sklearn.ensemble import GradientBoostingClassifier
sgd = GradientBoostingClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# # Model Evaluation

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


output = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
output.to_csv('submission.csv', index=False)

