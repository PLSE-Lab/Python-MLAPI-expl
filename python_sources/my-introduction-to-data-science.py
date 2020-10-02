#!/usr/bin/env python
# coding: utf-8

# # Data Science Introduction

# I am relatively new to data science, this is my workings for my first Kaggle submission. I hope that it is helpfull to other beginner, and of course any comments or improvements would be greatly appreciated.

# In[ ]:


# iports that we need:
# plotting:
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# data analysis
import numpy as np
import pandas as pd

#ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# so we can import data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# load the data
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# First, we will have a quick look at iour data, and make our first predictions.

# In[ ]:


# first look:
train.describe(include='all')


# Some notes and predictions to start off: <br><br> There are 891 entries in our training set. Most features (columns) don't have any missing values, only Age, Embarked and Cabin. <br> Cabin only has 204 values, we might want to drop this later on. <br><br> My initial thought are that Sex, Pclass and Age will be the most important factors, with the following effects:<br> - Females are more likely to survive<br> - Higher 'Passenger class' (i.e. 1st class) passengers will be more likely to survive<br> - Young childern will have a very high rate of survival

# # Analysing the data:

# ### Numerical Variables

# First, with a heatmap we can see how all of the numerical variables correlate with each other.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(train.corr(), annot=True, linewidth=0.5, fmt='.2f', ax=ax)
ax.set_ylim(7,0);


# Most of the feature variables don't correlate much with each other, which is good. There are a few exceptions, noticably Pclass with survived and Fare, and Parch with SibSp. It might be worth combining some of these features.

# ## Categorical Features:

# First, I'll make a function that will plot our categorical features against a numerical feature, to make visualisation easier.

# In[ ]:


def make_plot(x, y='Survived', df=train):
    sns.barplot(x=x, y=y, data=df)


# ### Sex

# In[ ]:


make_plot('Sex')


# We see that, as expected, females are more likely to survive. But it is (at least to me) slightly surprising how big this difference is. Sex will definitely be a very important variable to use

# ### Pclass

# In[ ]:


make_plot('Pclass')


# As predicted, the better the ticket the higher the chance of survival. Note this is a cetegorical feature disguising itself as a numerical feature - as there is an order, i.e. 1st class > 2nd class > 3rd class, this is acceptable

# ### Parents and Children, and Siblings and Spouses

# In[ ]:


make_plot('Parch')


# In[ ]:


make_plot('SibSp')


# Notice that there is a large amount of varience for larger numbers. I think this is bacause there is less data for these. Also, most groups wouldn't all be the same gender, making their chances of survival quite variable. Nevertheless, this does tell us information and there is a definite pattern.

# ### Age

# Age is a bit more difficult, as this is a quasi-continous feature. One way is to make bins, i.e. group ages together.

# In[ ]:


# making the bins:
bins = [-np.inf, 0, 5, 12, 18, 25, 35, 55, np.inf]
labels = ['unknown', 'infant', 'child', 'teenager', 'student', 'young_adult', 'adult', 'old']

for data in [train, test]: # note I do this to the test set too
    data['AgeGrp'] = pd.cut(data['Age'], bins=bins, labels=labels)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,4))
sns.barplot(x="AgeGrp", y='Survived', hue="Sex", data=train, ax=ax);
# survival accounting for gender


# In[ ]:


sns.barplot(x="AgeGrp", y='Survived', data=train);
# total survival


# As expected, infants had by far the highest total survival rates, but the top graph tells an interesting story. The female survival rate stays almost the same, except for children. But the male survival rate is very high for infants, and decreases quickly as age increases. So the main factor in the variatoin of survival rate are the male pasengers.

# # Missing Values

# Cleaning the data, and removing / guessing at the missing values.

# In[ ]:


pd.DataFrame(train.isnull().sum(), columns=['Train']).join(
                pd.DataFrame(test.isnull().sum(), columns=['Test'])
                )


# ### Missing value: fare

# In[ ]:


test[test['Fare'].isnull()]


# From the heatmap at the start, fare correlates well with passenger class. So we'll make a guess at his fare, from the median fare for his passenger class:

# In[ ]:


median_fare = test.groupby(['Pclass']).Fare.median()[3] # as he was in 3rd class
test['Fare'].fillna(median_fare, inplace=True)


# ### Missing values: embarked

# In[ ]:


train[train['Embarked'].isnull()]


# It's more difficult to find correlation with categorical data.

# In[ ]:


sns.countplot(x='Embarked', data=train);


# From the graph we can see that most passengers embarked at Southampton.

# In[ ]:


train['Embarked'] = train['Embarked'].fillna('S')


# ### Missing Values: Age

# There are a lot of missing values here, so we can't do these by hand. For simplicity, setting each value to the median of their Pclass and SibSp<br> (from the heatmap above, these are the two most correlated numerical variables)

# In[ ]:


train['Age'] = train.groupby(['Pclass', 'SibSp'])['Age'].apply(lambda x: x.fillna(x.median()))
train[train['Age'].isnull()]


# We still have some NAN values, because there are no age values for all members of the group Pclass=3 & SibSp=8. We can fill these in manually:

# In[ ]:


train['Age'].fillna(11, inplace=True)
train['Age'].isnull().sum()


# We do the same for the test set:

# In[ ]:


test['Age'] = test.groupby(['Pclass', 'SibSp'])['Age'].apply(
    lambda x: x.fillna(x.median()))
test['Age'].isnull().sum()


# ### Missing Values: Cabin

# Cabin has almost no values in it. I had a quick google to see why, and this is because the only cabin log retrieved that was legible had information on the more luxury cabins. This means that we could turn this into a boolean variable, but this would correlate a lot with Pclass so instead I will delete it. We can also delete PassengerId because this by definition has no correlaion with survival. Also, we can get rid of our Age Group column, which we used earlier to help with visualisation, as an actual age will have far more information.

# In[ ]:


for data in [train, test]:
    for feature in ['PassengerId', 'Cabin', 'AgeGrp']:
        data.drop(feature, inplace=True, axis=1)


# In[ ]:


train.head()


# ## Looking at the Name variable:

# I would have dropped this variable, but reading the many notebooks there is information to be found here:

# In[ ]:


for dataset in [train, test]:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# code from https://www.kaggle.com/startupsci/titanic-data-science-solutions    

pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


# grouping the uncommon names:
train['Title'] = train['Title'].replace(['Ms', 'Mlle'],'Miss')
train['Title'] = train['Title'].replace(['Mme'],'Mrs')
train['Title'] = train['Title'].replace(['Dr','Rev','the','Jonkheer','Lady','Sir', 'Don', 'Countess'],'Nobles')
train['Title'] = train['Title'].replace(['Major','Col', 'Capt'],'Navy')
train.Title.value_counts()


# In[ ]:


sns.barplot(x = 'Title', y = 'Survived', data=train);


# In[ ]:


# and for the tesst data - not all are present:

test['Title'] = test['Title'].replace(['Ms','Dona'],'Miss')
test['Title'] = test['Title'].replace(['Dr','Rev'],'Nobles')
test['Title'] = test['Title'].replace(['Col'],'Navy')
test.Title.value_counts()


# ## Encoding the features:

# To use machine learning, we have to have numerical not categorical freaures. There are two ways to do this, either:<br> - Replace categories by numbers<br> - One-hot encoding<br><br>I will use one-hot encoding as none of the following should be 'ordered', in the sense that you can't say:<br>embarked at Southampton > embarked at Cherbourg

# In[ ]:


categorical_features = [ 'Sex', 'Title', 'Embarked']

for feature in categorical_features:
    dummies = pd.get_dummies(train[feature]).add_prefix(feature+'_')
    train = train.join(dummies)

for feature in categorical_features:
    dummies = pd.get_dummies(test[feature]).add_prefix(feature+'_')
    test = test.join(dummies)


# We can then drop the original features:

# In[ ]:


for data in [train, test]:
    for feature in ['Name', 'Sex', 'Title', 'Embarked', 'Ticket']:
        data.drop(feature, axis=1, inplace=True)


# # Model Selection:

# First, we will split the training data so we can check how our models do.

# In[ ]:


# independant and dependant variables:
X = train.drop('Survived', axis=1)
y = train['Survived']


# In[ ]:


# this will split the model when we want to check our model.
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.22, random_state = 0)


# This is a small function to give us the accuracy of a given model:

# In[ ]:


from sklearn.metrics import accuracy_score


# I haven't scaled the data yet, this is next up:

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
test = sc.transform(test)


# Now, we will try several types of machine learning algorithm to try and find the best. I am using Grid Search to find which values to use for model hyperparameters (these are values that we need to input). Because of this, the grid search code below takes a long time to run, especially on kaggle.

# In[ ]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_val)
round(accuracy_score(y_pred, y_val) * 100, 2)


# In[ ]:


# decision tree:
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold


#Here we will use gridsearchcv to find the best values for our hyperparameter
# kfold is for its internal validation:
cv = KFold(n_splits=10, shuffle=True, random_state=42)

params = dict(max_depth=range(1,10),
              max_features=[2, 4, 6, 8],
              criterion=['entropy', 'gini']
             )
DTGrid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    param_grid=params, verbose=False,
                    cv=cv)

DTGrid.fit(X_train, y_train)
DecTree = DTGrid.best_estimator_
print(DTGrid.best_params_)
round(DecTree.score(X_val, y_val) * 100, 2)


# In[ ]:


# random forest, using grid search as above:
from sklearn.ensemble import RandomForestClassifier

cv=KFold(n_splits=10, shuffle=True, random_state=42)

params = {'n_estimators': [80, 100, 120, 140],
              'max_depth': range(2,7),
              'criterion': ['gini', 'entropy']      
        }


RFGrid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid=params, verbose=False,
                    cv=cv)

RFGrid.fit(X_train, y_train)
RandForest = RFGrid.best_estimator_
print(RFGrid.best_params_)
round(RandForest.score(X_val, y_val) * 100, 2)


# In[ ]:


# k nearest neighbors:
from sklearn.neighbors import KNeighborsClassifier

params = dict(n_neighbors=[3,6,8,10],
              weights=['uniform', 'distance'],
              metric=['euclidean', 'manhattan']
              )
cv=KFold(n_splits=10, shuffle=True, random_state=42)

KNNGrid = GridSearchCV(KNeighborsClassifier(),
                    param_grid=params, verbose=False,
                    cv=cv)

KNNGrid.fit(X_train, y_train)
KNN = KNNGrid.best_estimator_
print(KNNGrid.best_params_)
round(KNN.score(X_val, y_val) * 100, 2)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.score(X_val, y_val)


# In[ ]:


# Support Vector Machines:
from sklearn.svm import SVC

params = {'C':[100,500,1000],
          'gamma':[0.1,0.001,0.0001],
          'kernel':['linear','rbf']
          }
cv=KFold(n_splits=10, shuffle=True, random_state=42)

SVMGrid = GridSearchCV(SVC(random_state=42),
                    param_grid=params, verbose=False,
                    cv=cv)

SVMGrid.fit(X_train, y_train)
SVM = SVMGrid.best_estimator_
print(SVMGrid.best_params_)
round(SVM.score(X_val, y_val) * 100, 2)


# In[ ]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

params = {'loss':['hinge', 'perceptron'],
          'alpha':[0.01, 0.001, 0.0001],
          'penalty':['l2', 'l1']
          }
cv=KFold(n_splits=10, shuffle=True, random_state=42)

SGDGrid = GridSearchCV(SGDClassifier(random_state=42),
                    param_grid=params, verbose=False,
                    cv=cv)

SGDGrid.fit(X_train, y_train)
SGD = SGDGrid.best_estimator_
print(SGDGrid.best_params_)
round(SGD.score(X_val, y_val) * 100, 2)


# In[ ]:


# gradient Boosting:
from sklearn.ensemble import GradientBoostingClassifier

params = {'loss':[ 'deviance', 'exponential'],
          'learning_rate':[ 0.1, 0.01, 0.001],
          'n_estimators':[100, 400, 700]
          }

cv=KFold(n_splits=10, shuffle=True, random_state=42)

GBGrid = GridSearchCV(GradientBoostingClassifier(random_state=42),
                    param_grid=params, verbose=False,
                    cv=cv)

GBGrid.fit(X_train, y_train)
GB = GBGrid.best_estimator_
print(GBGrid.best_params_)
round(GB.score(X_val, y_val) * 100, 2)


# The best predictor was Stochastic Graidient Descent, so we use this to predict our values. Note we will fit the entire training set, to get the best predictions:

# In[ ]:


model = SGDClassifier(**SGDGrid.best_params_)
model.fit(X, y)

test_id = pd.read_csv('../input/titanic/test.csv')
submission = pd.DataFrame({'PassengerId': test_id['PassengerId'], 'Survived': model.predict(test) })
submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)
submission.to_csv('submission.csv', index=False)


# This has an accuracy score of 78.4%, which is not bad for a first attempt!
