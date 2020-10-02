# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rnd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
# for visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')



# %% [code]
print(train_df.columns.values)

# %% [code]
train_df.head()

# %% [code]
train_df.tail()

# %% [code]
train_df.info()
print("--------------------------------------------------------------")
test_df.info()

# %% [code]
train_df.describe(percentiles=[.1,.2,.3,.4,.61, .62] )

# %% [code]
train_df.Name.nunique()

# %% [code]
train_df.Fare.nunique()

# %% [code]
train_df.describe(include='O')

# %% [code]
# pivot features to know better correlation
train_df[['Pclass', 'Survived']]. groupby('Pclass', as_index= False).mean().sort_values(by='Survived', ascending= False)

# %% [code]
train_df[['Sex', 'Survived']].groupby('Sex', as_index = False).mean().sort_values(by='Survived', ascending = False)

# %% [code]
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index= False).mean().sort_values('Survived', ascending = False)

# %% [code]
train_df[['Parch', 'Survived']].groupby('Parch', as_index = False).mean().sort_values(by = 'Survived', ascending = False)

# %% [code]
g = sns.FacetGrid(train_df, col = 'Survived')
g.map(plt.hist, 'Age', bins=20)

# %% [code]
grid = sns.FacetGrid(train_df, col = 'Survived', row='Pclass', height=2.5, aspect = 2)
grid.map(plt.hist, 'Age', alpha = 0.6, bins= 20)
grid.add_legend()

# %% [code]
# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

# %% [code]
grid = sns.FacetGrid(train_df, row ='Embarked', col = 'Survived',height = 2.5, aspect = 2)
grid.map(sns.barplot, 'Sex', 'Fare',ci = None)
 

# %% [code]
# dropping columns that are not required
train_df = train_df.drop(['Ticket', 'Cabin'], axis = 1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis = 1)
# new shape
print(train_df.shape, test_df.shape)

# %% [code]
train_df.head()

# %% [code]
both = [train_df, test_df]
for dataset in both:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
pd.crosstab(train_df['Title'], train_df['Sex'])

# %% [code]
# replace many titles with one word
for dataset in both:
    dataset["Title"] = dataset['Title'].replace(['Lady', 'Countess',\
                                                 'Capt', 'Col', 'Don', 'Dr', 'Major',\
                                                'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train_df[['Title', 'Survived']].groupby(['Title'], as_index = False).mean()

# %% [code]
train_df.head()

# %% [code]
#converting categorial titles to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
print(train_df.head())
combine = [train_df, test_df]
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
    

# %% [code]
# drop name and passenger id from dataset
train_df = train_df.drop(['Name', 'PassengerId'], axis = 1)
test_df = test_df.drop(['Name'], axis = 1)
both = [train_df, test_df]
train_df.shape, test_df.shape

# %% [code]
# convert 'sex' feature to hold numerical value
for dataset in both:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
    
train_df.head()    


# %% [code]
# completing featureswith  missing values
# fill ages with median across pclass and gender combinations

grid = sns.FacetGrid(train_df, row= 'Pclass', col='Sex', height = 2.5, aspect= 2)
grid.map(plt.hist, 'Age', bins = 20)


# %% [code]
#empty array for guessed ages
guess_ages = np.zeros((2,3))
guess_ages

# %% [code]
for dataset in both:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j+1)]['Age'].dropna()
            
            age_guess = guess_df.median()
            
            guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5
#     print(guess_ages)        
    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age']\
            = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)
            
train_df.head()            
            

# %% [code]
# create age bands to find correlation with survived

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending  = True)

# %% [code]
# give age column ordinals based on agebands we made
for dataset in both:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()

# %% [code]
# drop ageband now that we have given corresponding ordinals to age

train_df = train_df.drop(['AgeBand'], axis = 1)
both = [train_df, test_df]
train_df.head()

# %% [code]
# create new feature family size
for dataset in both:
    dataset['FamilySize'] = dataset['SibSp'] + dataset["Parch"] + 1
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index= False).mean().sort_values(by='Survived', ascending=False)

# %% [code]
# MAKE other feature isAlone
for dataset in both:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index= False).mean().sort_values(by='Survived', ascending=False)

# %% [code]
# now we can drop Parch, SibSp, familySize
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)

# %% [code]
both = [train_df, test_df]
train_df.head()

# %% [code]
# artficial feature class*age
for dataset in both:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
    
train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head()    

# %% [code]
#completing categorial feature : embarkation by replacing with most common occurence
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port

# %% [code]
for dataset in both:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending= False)    

# %% [code]
# convert embarked feature to numeric type
for dataset in both:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C': 1, 'Q': 2}).astype(int)
train_df.head()    

# %% [code]
# complete field by filling the single null value
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace = True)
test_df.head()

# %% [code]
# create fareband and then assign integers based on fareband
# cut creates eqally spaced bands while qcut makes band which have equal frequency of members
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='Survived', ascending=True)

# %% [code]
for dataset in both:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop(['FareBand'], axis = 1)
both = [train_df, test_df]

train_df.head()

# %% [code]
test_df.head()

# %% [code]
# now we start actual training
# logistic regression
X_train = train_df.drop('Survived', axis = 1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis = 1)
X_train.shape, Y_train.shape, X_test.shape

# %% [code]
lreg = LogisticRegression(solver = 'lbfgs')
lreg.fit(X_train, Y_train)
Y_pred = lreg.predict(X_test)
accu = lreg.score(X_train, Y_train)
accu

# %% [code]
coeffs = pd.DataFrame(train_df.columns.delete(0))
coeffs.columns = ['Feature']
coeffs['Correlation'] = pd.Series(lreg.coef_[0])

coeffs.sort_values(by='Correlation', ascending = False)


# %% [code]
# support vector machine

svc = SVC(gamma = 'auto')
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
acc_svc = svc.score(X_train, Y_train)
acc_svc

# %% [code]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = knn.score(X_train, Y_train)
acc_knn

# %% [code]
#naive bayes
nb = GaussianNB()
nb.fit(X_train, Y_train)
nb.predict(X_test)
acc_nb = nb.score(X_train, Y_train)
acc_nb

# %% [code]
# perceptron

pr = Perceptron()
pr.fit(X_train, Y_train)
pr.predict(X_test)
acc_pr = pr.score(X_train, Y_train)
acc_pr

# %% [code]
#linearSVC

lin_SVC = LinearSVC(max_iter=1000)
lin_SVC.fit(X_train, Y_train)
lin_SVC.predict(X_test)
acc_linSVC = lin_SVC.score(X_train, Y_train)
acc_linSVC

# %% [code]
# stocastic gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
sgd.predict(X_test)
acc_sgd = sgd.score(X_train, Y_train)
acc_sgd

# %% [code]
# decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
dt.predict(X_test)
acc_dt = dt.score(X_train, Y_train)
acc_dt

# %% [code]
#RandomForest

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
rf.predict(X_test)
acc_rf = rf.score(X_train, Y_train)
acc_rf


# %% [code]
# rank models
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machine', 'KNN', 'Random Forest', 'LinearSVC', 'SGD', 'Decision Tree', 'Naive Bayes', 'Perceptron'],
    'Score': [accu, acc_svc,acc_knn, acc_rf, acc_linSVC, acc_sgd, acc_dt, acc_nb, acc_pr]
})
models.sort_values(by='Score', ascending = False)

# %% [code]
Y_pred = dt.predict(X_test)

# %% [code]
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": Y_pred
})
submission.head()

# %% [code]
submission.to_csv('TitanicSubmission1.csv', index = False)
# submission.to_csv('../output/Titanicsubmission1.csv', index=False)