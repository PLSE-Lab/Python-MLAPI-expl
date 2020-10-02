#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import random as rnd


# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
# train_df
test_df = pd.read_csv('../input/titanic/test.csv')
# test_df
combine = [train_df, test_df]


# In[ ]:


print(train_df.columns.values)


# In[ ]:


train_df.head()


# In[ ]:


train_df.tail()


# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# In[ ]:


train_df.describe()


# In[ ]:


train_df.describe(include = ['O'])


# In[ ]:


a = train_df[["Sex", "Survived"]].groupby(['Sex']).mean()
a


# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


train_df[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


train_df[['Embarked', 'Fare', 'Sex']].groupby(['Embarked'], as_index = False).mean().sort_values(by = 'Fare', ascending = False)


# In[ ]:


train_df[['Parch', 'Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


g = sns.FacetGrid(train_df, col ='Survived')
g.map(plt.hist, "Age", bins=20)


# In[ ]:


grid = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass', height = 2.2, aspect = 1.6)
grid.map(plt.hist, 'Age', bins=20, alpha = 0.5)
grid.add_legend()


# In[ ]:


grid = sns.FacetGrid(train_df, row = 'Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = 'deep')
grid.add_legend()


# In[ ]:


grid = sns.FacetGrid(train_df, row = 'Embarked', col = 'Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha = 0.5, ci = None)
grid.add_legend()


# In[ ]:


print( "Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape )

train_df = train_df.drop(['Ticket', 'Cabin'], axis = 1)


# In[ ]:


train_df.shape
test_df = test_df.drop(['Ticket', 'Cabin'], axis = 1)
combine = [train_df, test_df]


# In[ ]:


print( "Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape )


# In[ ]:



for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)


# In[ ]:


pd.crosstab(train_df['Title'], train_df['Sex'])


# In[ ]:


# train_df[['Title', 'Survived']].groupby('Title').mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms', 'Mme'], 'Miss')


# In[ ]:


train_df[['Title', 'Survived']].groupby('Title').mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    


# In[ ]:


train_df.head()


# In[ ]:


train_df = train_df.drop(['Name', 'PassengerId'], axis = 1)


# In[ ]:


test_df = test_df.drop(['Name'], axis = 1)


# In[ ]:


train_df.shape, test_df.shape


# In[ ]:


train_df[['Sex', 'Survived']].groupby('Sex').mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


train_df.info()


# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({"female": 1, "male": 0})


# In[ ]:


combine[0].head()


# In[ ]:


train_df['Sex'] = train_df['Sex'].map({"female": 1, "male": 0})


# In[ ]:


test_df['Sex'] = test_df['Sex'].map({"female": 1, "male": 0})


# In[ ]:


test_df.head()


# In[ ]:


train_df.info()


# In[ ]:


grid = sns.FacetGrid(train_df, row = 'Pclass', col = 'Sex', height = 2.2, aspect = 1.6)
grid.map(plt.hist, 'Age', bins = 20)
grid.add_legend()


# In[ ]:


guess_ages = np.zeros((2,3))
guess_ages


# In[ ]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


# In[ ]:


train_df.head()


# In[ ]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', "Survived"]].groupby(['AgeBand'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


train_df.head()


# In[ ]:


for dt in combine:
    dt.loc[dt['Age'] <= 16, 'Age' ] = 0
    dt.loc[(dt['Age'] > 16) & (dt['Age'] <= 32), 'Age' ] = 1
    dt.loc[(dt['Age'] > 32) & (dt['Age'] <= 48), 'Age' ] = 2
    dt.loc[(dt['Age'] > 48) & (dt['Age'] <= 64), 'Age' ] = 3
    dt.loc[dt['Age'] > 64, 'Age' ] = 4
    
    


# In[ ]:


combine[1].head()


# In[ ]:


train_df = combine[0]
combine = [train_df, test_df]
train_df.head()


# In[ ]:


train_df = combine[0].drop(['Name', 'PassengerId'], axis=1)
combine = [train_df, test_df]
train_df.head()


# In[ ]:


for dt in combine:
    dt['FamilySize'] = dt['SibSp'] + dt['Parch'] + 1


# In[ ]:


train_df[['FamilySize', "Survived"]].groupby(['FamilySize'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[ ]:




train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()


# In[ ]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass


# In[ ]:


train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# In[ ]:


freq = train_df.Embarked.dropna().mode()[0]
freq


# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )


# In[ ]:


train_df.head()


# In[ ]:



train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace = True)


# In[ ]:


test_df.head()


# In[ ]:




train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:




for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


# In[ ]:


combine[1].head()


# In[ ]:


test_df.head(10)


# In[ ]:


df = combine[1]
df[df.isnull().any(axis=1)]


# In[ ]:


X_train1 = combine[0].drop('Survived', axis = 1)
X_train2 = X_train1.drop('Age', axis = 1)
X_train = X_train2.drop('Age*Class', axis = 1)

Y_train = combine[0]['Survived']
X_test1 = combine[1].drop(['PassengerId'], axis = 1).copy()
X_test2 = X_test1.drop('Age', axis = 1).copy()
X_test = X_test2.drop(['Age*Class'], axis = 1).copy()


# In[ ]:


combine[1].head()


# In[ ]:


X_train.shape, Y_train.shape, X_test


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


logreg =  LogisticRegression()
logreg.fit(X_train, Y_train)


# In[ ]:


Y_pred = logreg.predict(X_test)


# In[ ]:


acc_log = round(logreg.score(X_train, Y_train)*100, 2)


# In[ ]:


acc_log


# In[ ]:


logreg.score(X_train, Y_train)


# In[ ]:





# In[ ]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))


# In[ ]:


train_df


# In[ ]:


coeff_df


# In[ ]:


coeff_df.columns = ['Feature']


# In[ ]:


coeff_df


# In[ ]:


coeff_df["Correlation"] = pd.Series(logreg.coef_[0])


# In[ ]:


logreg.coef_


# In[ ]:


coeff_df.sort_values(by = 'Correlation', ascending = False)


# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)


# In[ ]:


Y_pred = svc.predict(X_test)


# In[ ]:


acc_svc = round(svc.score(X_train, Y_train)*100, 2)


# In[ ]:


acc_svc


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)


# In[ ]:


knn.fit(X_train ,Y_train)


# In[ ]:


Y_pred = knn.predict(X_test)


# In[ ]:


acc_knn = round(knn.score(X_train, Y_train)*100, 2)


# In[ ]:


acc_knn


# In[ ]:



gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:



perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:



linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[ ]:



decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


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


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })


# In[ ]:


submission


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




