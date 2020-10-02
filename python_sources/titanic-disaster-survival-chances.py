#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster

# ### Description: 
# 
#     The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
#     One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
#     In this challenge, complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# ### Import necessery libraries: 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os


# ### Import Data

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv(os.path.join(dirname, 'train.csv'), sep=",")
train_data.head()


# In[ ]:


test_data = pd.read_csv(os.path.join(dirname, 'test.csv'), sep=",")
test_data.head()


# ### EDA

# In[ ]:


test_data.shape


# In[ ]:


train_data.shape


# In[ ]:


train_data.info()


# In[ ]:


# ===> unique values count:
train_data.PassengerId.nunique()


# In[ ]:


train_data.Survived.value_counts()


# In[ ]:


train_data.Pclass.value_counts()


# In[ ]:


train_data.Sex.value_counts()


# In[ ]:


train_data.Cabin.value_counts().head(10)
print(train_data[train_data.Cabin.isnull() == False]['Pclass'].value_counts())
print(train_data[train_data.Cabin.isnull() == False]['Survived'].value_counts())


# In[ ]:


train_data.Pclass.value_counts()


# In[ ]:


train_data.Sex.value_counts()


# In[ ]:


train_data.Ticket.nunique()


# #### Add co passengers to the user:
#     Based on the ticket number we can group the user and attain information as in how many co-passengers exists with the passenger. Family generally will have the same ticket id generated. 

# In[ ]:


t = train_data.Ticket.value_counts()
train_data['TicketCoPassengers'] = train_data.Ticket.apply(lambda x: t[x]-1) 
train_data.head()


# In[ ]:


t = test_data.Ticket.value_counts()
test_data['TicketCoPassengers'] = test_data.Ticket.apply(lambda x: t[x]-1)


# In[ ]:


train_data.Ticket.value_counts().head(10)


# In[ ]:


train_data[train_data.Ticket == '110152']


# In[ ]:


train_data.groupby('Ticket').TicketCoPassengers.count().head()


# In[ ]:


train_data.head()


# #### Check for Null values: 

# In[ ]:


train_data.isnull().sum(axis=0) / train_data.shape[0]


# In[ ]:


train_data.describe(include=['O'])


# In[ ]:


train_data.Age.value_counts().head()


# ### Univariate analysis: 

# In[ ]:


sns.distplot(train_data.Fare.dropna(), kde=False, bins=30)


# #### No.of co-passengers:

# In[ ]:


sns.distplot(train_data.TicketCoPassengers.dropna(), kde=False, bins=30)


# In[ ]:


sns.countplot(train_data.Embarked.fillna('NoClass'))


# #### Survival rate:

# In[ ]:


sns.countplot(train_data.Survived)


# #### Family siblings:

# In[ ]:


sns.countplot(train_data.Parch) 


# ### Bi-variate analysis: 

# In[ ]:


sns.countplot(hue='Survived', x='Parch', orient='h', data=train_data, )


# In[ ]:


train_data.info()


# #### Embarked survival rate: 

# In[ ]:


sns.countplot(x=train_data.Embarked, hue=train_data.Survived)


# #### Survival chances based on class:

# In[ ]:


sns.countplot(x='Pclass', hue='Survived', data=train_data)


# Note: Can see that First Class passengers has more survival chances.

# #### Survival chances based on gender:

# In[ ]:


sns.countplot(x='Sex', hue='Survived', data=train_data)


# Note: Female has more survival chances than men.

# In[ ]:


plt.figure()
sns.distplot(train_data[train_data.Survived == 1].Age.dropna(), bins=30)
sns.distplot(train_data[train_data.Survived == 0].Age.dropna(), bins=30)
plt.show()


# Note: Survival rate of age below 20 is more compared to other age group

# ### Feature engineering:

# #### Age column: 

# In[ ]:


g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


train_data.Age = train_data.Age.fillna(train_data.Age.mean())
test_data.Age = test_data.Age.fillna(test_data.Age.mean())


# In[ ]:


train_data.Age.isnull().sum(axis=0)


# In[ ]:


bins = [0, 15, 40, 60, 120]
labels = [1, 2, 3, 4]
train_data['age_bin'] = pd.cut(train_data.Age, bins=bins, labels=labels)
test_data['age_bin'] = pd.cut(test_data.Age, bins=bins, labels=labels)


# In[ ]:


train_data.head()


# #### Fare column: 

# In[ ]:


plt.figure()
g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Fare', bins=20)
plt.show()


# In[ ]:


train_data.Fare = train_data.Fare.fillna(train_data.Fare.mean())
test_data.Fare = test_data.Fare.fillna(test_data.Fare.mean())


# In[ ]:


bins = [-1, 50, 100, 200, 10000]
labels = [1, 2, 3, 4]
train_data['Fare_bins'] = pd.cut(train_data.Fare, bins=bins, labels=labels)
test_data['Fare_bins'] = pd.cut(test_data.Fare, bins=bins, labels=labels)


# In[ ]:


train_data.head()


# #### Embarked column:

# In[ ]:


train_data.Embarked.value_counts()


# In[ ]:


train_data[train_data.Embarked.isnull()]


# In[ ]:


train_data.Embarked = train_data.Embarked.fillna('S')
test_data.Embarked = test_data.Embarked.fillna('S')


# In[ ]:


train_data.head()


# In[ ]:


train_data.Sex = train_data.Sex.map({'male': 1, 'female': 0})
test_data.Sex = test_data.Sex.map({'male': 1, 'female': 0})


# In[ ]:


# ===> 
train_data = pd.get_dummies(train_data, columns=['Pclass', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Pclass', 'Embarked'], drop_first=True)
train_data.head()


# In[ ]:


train_data.drop(columns=['Age', 'Fare', 'Ticket'], inplace=True)
test_data.drop(columns=['Age', 'Fare', 'Ticket'], inplace=True)


# #### Title Column:

# In[ ]:


train_data.loc[:, 'Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data.loc[:, 'Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


pd.crosstab(train_data.Title, train_data.Survived)


# In[ ]:


def replaceTitles(_data):
    _data['Title'] = _data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    _data['Title'] = _data['Title'].replace('Mlle', 'Miss')
    _data['Title'] = _data['Title'].replace('Ms', 'Miss')
    _data['Title'] = _data['Title'].replace('Mme', 'Mrs')
    return _data

train_data = replaceTitles(train_data)
test_data = replaceTitles(test_data)
print(train_data['Title'].value_counts())
print(test_data['Title'].value_counts())


# In[ ]:


train_data = pd.get_dummies(train_data, columns=['Title'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Title'], drop_first=True)


# In[ ]:


train_data.drop(columns=['Name'], inplace=True)
test_data.drop(columns=['Name'], inplace=True)


# In[ ]:


train_data.head()


# In[ ]:


#====> Fill cabin NaN as not assigned
train_data.drop(columns=['Cabin', 'PassengerId'], inplace=True)
test_data_passengers = test_data.PassengerId
test_data.drop(columns=['Cabin', 'PassengerId'], inplace=True)
# train_data.Cabin.fillna('NotAssigned').head()


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


x_train = train_data.drop(columns=['Survived'])
y_train = train_data.Survived


# ### Modeling: 

# **We can try different categorical models and try to see for best fit of the model**
# 
# * Logistic Regression
# * KNN or k-Nearest Neighbors
# * Support Vector Machines
# * Decision Tree
# * Random Forrest

# #### Logistic Regression: 

# In[ ]:


log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(test_data)
log_reg.score(x_train, y_train)


# In[ ]:


log_reg.coef_[0]


# In[ ]:


corr = pd.DataFrame({'features': x_train.columns, 'coff': log_reg.coef_[0]}).sort_values(by='coff', ascending=False)
corr


# In[ ]:


train_data.shape


# ##### GridsearchCV for parameter tuning: 

# In[ ]:


g_logit = LogisticRegression()
params = {
    "C": np.logspace(-3, 3, 7),
    "penalty": ["l1", "l2"]  # l1 lasso l2 ridge
}
kfold = KFold(n_splits=2, random_state=101, shuffle=True)

grid = GridSearchCV(g_logit,
                    param_grid=params,
                    n_jobs=-1,
                    verbose=True,
                    return_train_score=True,
                    scoring='accuracy',
                    cv=kfold)
grid.fit(x_train, y_train)


# In[ ]:


grid.best_score_


# In[ ]:


results = pd.DataFrame(grid.cv_results_)
results.sort_values(by='mean_test_score', ascending=False)


# In[ ]:


logit_alog = grid.best_estimator_
logit_alog.fit(x_train, y_train)
logit_y_predict = logit_alog.predict(test_data)
print("Accuracy score for the logistic regression::", logit_alog.score(x_train, y_train)*100)


# ### KNN algorithm: 

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=4, n_jobs=-1)
knn.fit(x_train, y_train)
knn_y_pred = knn.predict(test_data)


# In[ ]:


knn.score(x_train, y_train)


# In[ ]:


knn.get_params


# ##### GridsearchCV for parameter tuning: 

# In[ ]:


g_knn = KNeighborsClassifier()
params = {
    'n_neighbors': range(2, 20),
    'leaf_size': range(4, 40, 4)
}

kfold = KFold(n_splits=2, shuffle=True, random_state=101)

grid = GridSearchCV(g_knn,
                    param_grid=params,
                    cv=kfold,
                    n_jobs=-1,
                    scoring='accuracy',
                    verbose=True,
                    return_train_score=True)
grid.fit(x_train, y_train)


# In[ ]:


grid.best_estimator_


# In[ ]:


grid.best_score_


# In[ ]:


grid.best_params_


# In[ ]:


results = pd.DataFrame(grid.cv_results_)
results.sort_values(by='mean_test_score', ascending=False)


# Note: Choose the best model with the train and test accuary.

# In[ ]:


knn_algo = grid.best_estimator_
knn_algo.fit(x_train, y_train)
knn_y_pred = knn.predict(test_data)
print("Accuracy score for the logistic regression::", knn_algo.score(x_train, y_train)*100)


# ### Support Vector Machines: 

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svc_alog = SVC()
params = {
    'C': [0.1, 1, 10, 20, 30, 40, 50, 100, 150],
    'kernel': ['rbf']
}
svc_grid = GridSearchCV(svc_alog,
                        param_grid=params,
                        cv=kfold,
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=True,
                        return_train_score=True)
svc_grid.fit(x_train, y_train)


# In[ ]:


results = pd.DataFrame(svc_grid.cv_results_)
results.sort_values(by='mean_test_score', ascending=False)


# In[ ]:


svc_grid.best_score_


# In[ ]:


svc_alog = svc_grid.best_estimator_
svc_alog.fit(x_train, y_train)
svc_y_pred = svc_alog.predict(test_data)
print("Linear SVM algorithm accuracy score:: ", svc_alog.score(x_train, y_train)*100) 


# ### Decision Tree algorithm:

# In[ ]:


decision_alog = DecisionTreeClassifier()


# In[ ]:


decision_alog.fit(x_train, y_train)
decision_alog.score(x_train, y_train)


# ##### GridsearchCV for parameter tuning

# In[ ]:


decision_alog


# In[ ]:


g_decision_alog = DecisionTreeClassifier()
params = {
    'min_samples_split': range(2, 20, 2),
    'max_leaf_nodes': range(2, 10, 2)
}
decision_tree_grid = GridSearchCV(g_decision_alog,
                                  param_grid=params,
                                  verbose=True,
                                  scoring='accuracy',
                                  return_train_score=True,
                                  cv=kfold,
                                  n_jobs=-1)
decision_tree_grid.fit(x_train, y_train)


# In[ ]:


decision_tree_grid.best_score_


# In[ ]:


decision_tree_grid.best_estimator_


# In[ ]:


result = pd.DataFrame(decision_tree_grid.cv_results_)
result.sort_values(by='mean_test_score', ascending=False)


# In[ ]:


decision_tree_algo = decision_tree_grid.best_estimator_
decision_tree_algo.fit(x_train, y_train)
dec_tree_y_pred = decision_tree_algo.predict(test_data)
print("Decision Tree alogirthm score:: ", decision_tree_algo.score(x_train, y_train) * 100) 


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


decision_tree_algo


# In[ ]:


d_alog = DecisionTreeClassifier(max_leaf_nodes=8, min_samples_split=2)
adaboost = AdaBoostClassifier(d_alog, n_estimators=200)
# adaboost.fit(x_train, y_train)


# In[ ]:


kfold = KFold(n_splits=2, random_state=101, shuffle=True)
algo = DecisionTreeClassifier(max_leaf_nodes=8, min_samples_split=2)
adaboost = AdaBoostClassifier(base_estimator=algo, n_estimators=200)
params = {
    "base_estimator__criterion": ["gini", "entropy"],
    "base_estimator__splitter": ["best", "random"],
    "n_estimators": range(2, 200, 2)
}
adaboost_grid = GridSearchCV(adaboost,
                             param_grid=params,
                             verbose=True,
                             scoring='accuracy',
                             return_train_score=True,
                             cv=kfold,
                             n_jobs=-1)
adaboost_grid.fit(x_train, y_train)


# In[ ]:


adaboost_grid.best_score_


# In[ ]:


result = pd.DataFrame(adaboost_grid.cv_results_)
result.sort_values(by='mean_test_score', ascending=False)


# In[ ]:


adaboost_grid.best_params_


# ### Random forest:

# In[ ]:


rfc = RandomForestClassifier()
params = {
    'min_samples_split': range(2, 20, 2),
    'max_leaf_nodes': range(2, 10, 2)
}
rfc_grid = GridSearchCV(rfc,
                        param_grid=params,
                        verbose=True,
                        scoring='accuracy',
                        return_train_score=True,
                        cv=kfold,
                        n_jobs=-1)
rfc_grid.fit(x_train, y_train)


# In[ ]:


rfc_grid.best_score_


# In[ ]:


rfc_grid.best_params_


# ### Use Random forest as final model and fetch the result:

# In[ ]:


final_model = rfc_grid.best_estimator_
rfc_y_pred = final_model.predict(test_data)


# In[ ]:


submission = pd.DataFrame({'PassengerId': test_data_passengers, 'Survived': rfc_y_pred})
submission.head()


# In[ ]:


# submission.to_csv('../output/submission.csv', index=False)


# In[ ]:




