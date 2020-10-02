#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# # **Aquire data**

# In[ ]:


## Import data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# # **Exploratory data analysis**

# In[ ]:


# preview the data
train_df.head()


# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# *   Categorical feature: Survived, Sex, and Embarked. Ordinal: Pclass.
# *   Continous feature: Age, Fare. Discrete: SibSp, Parch.
# *   Notice that there is a lot of null value in Cabin(77% missing in trainig dataset, 78% missing in test dataset).
# *   Age and Embarked columns are incomplete in training dataset. Age feature is incomplete in test dataset.
# *   We decide to drop unnecessary columns(PassengerId, Name, Ticket) and Cabin columns for future analysis and prediction.
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


train_df.describe()


# In[ ]:


train_df.describe(include='O')


# *   Total samples are 891, which is 40% of the actual number of passengers on board(2,224).
# *   38.3% sample survival rate while the actual survival rate was 32.5%(722 out of 2224).
# 
# 

# ## Pclass
# 

# In[ ]:


sns.countplot(x='Pclass', data=train_df)
sns.factorplot('Pclass','Survived',order=[1,2,3], data=train_df, size=5)


# In[ ]:


train_df['Pclass'].value_counts()


# In[ ]:


print (train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())


# *   Over half of the passengers were in Class 3(55.1%).
# *   Upper-class(Pclass = 1) passengers were more likely to survive with  62.9% survival rate.
# *   Class 3 passengers only had 24.2% survival rate.
# 
# 
# 

# ## Sex

# In[ ]:


sns.countplot(x='Sex', data=train_df)


# In[ ]:


train_df['Sex'].value_counts()


# In[ ]:


sns.barplot(x='Sex', y='Survived', data=train_df)


# In[ ]:


print (train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())


# In[ ]:


## combine influence of sex and pclass
g = sns.FacetGrid(train_df, col="Sex")
g = g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')


# * 64.8% of the passengers are male.
# * Female passengers have a high survival rate(74.2%) while only 18.8% male passengers survived. 

# ## Age

# In[ ]:


## Understand the age distribution of passengers.
train_df['Age'].hist(bins=70)


# In[ ]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# *  Most passengers are in 15-40 age range.
# *  Few elderly passengers (<1%) within age range 65-80. The oldest passengers (Age = 80) survived.
# *  Children (Age <= 8) were more likely to have survived(Over 60% survival rate).
# * Age group 8-12 have the lowest survival rate(Under 30%).
# *  For female, they have a high survival rate regardless of age while male passengers have a pretty low survival rate expect for children.
# 
# 
# 
# 

# ## Family 

# In[ ]:


train_df['Parch'].value_counts()


# In[ ]:


train_df['SibSp'].value_counts()


# In[ ]:


train_df['Family'] =  train_df["Parch"] + train_df["SibSp"]
train_df['Family'].value_counts()


# In[ ]:


print (train_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean())


# * Most passengers did not travel with family members. Less than 25% of the passengers traveled with parents or children. Only 31.7% of the passengers had siblings and/or spouse aboard.
# * Passengers who travel with family member were more likely to survive.

# ## Fare

# In[ ]:


train_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100)


# In[ ]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# * Fares varied significantly with few passengers (<1%) paying as high as $512.
# * Higher fare paying passengers had better survival. This feature is similar to Pclass.
# 

# ## Embarked
# 

# In[ ]:


train_df['Embarked'].value_counts()


# In[ ]:


print (train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())


# In[ ]:


grid = sns.FacetGrid(train_df, col='Embarked', size=3, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# * Most passengers embarked at S(72.4%).
# * Male passengers who embarked at C had higher survival rate than female 
# * Ports of embarkation have varying survival rates for Pclass=3 among male passengers. 

# # Data Cleaning

# In[ ]:


## Drop unnecessary columns

train_df = train_df.drop(['PassengerId','Name','Ticket','Cabin','FareBand'], axis=1)
test_df    = test_df.drop(['Name','Ticket','Cabin'], axis=1)


# In[ ]:


## Sex

## Converting categorical feature Sex into numerical values where female = 1 and male = 0.
train_df.loc[train_df['Sex']=='male','Sex'] = 0
train_df.loc[train_df['Sex']=='female','Sex'] = 1
test_df.loc[test_df['Sex']=='male','Sex'] = 0
test_df.loc[test_df['Sex']=='female','Sex'] = 1


# In[ ]:


## Age

## generate random numbers between (mean - std) and (mean + std) to replace missing value.

average_age_train   = train_df["Age"].mean()
std_age_train      = train_df["Age"].std()
count_nan_age_train = train_df["Age"].isnull().sum()

average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

rand_1 = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train, size = count_nan_age_train)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

train_df["Age"][np.isnan(train_df["Age"])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

train_df['Age'] = train_df['Age'].astype(int)
test_df['Age']    = test_df['Age'].astype(int)


# In[ ]:


## Family

## Use family feature to replace Parch and SibSp.

train_df['Family'] =  train_df["Parch"] + train_df["SibSp"]
train_df['Family'].loc[train_df['Family'] > 0] = 1
train_df['Family'].loc[train_df['Family'] == 0] = 0

test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0

train_df = train_df.drop(['SibSp','Parch'], axis=1)
test_df    = test_df.drop(['SibSp','Parch'], axis=1)


# In[ ]:


## Embarked

## Fill the missing values with the most common occurrence(S).

train_df['Embarked'] = train_df['Embarked'].fillna('S')

## Convert the Embarked feature by creating a new numeric feature.

train_df['Embarked'] = train_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_df['Embarked'] = test_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


## Fare 

## There is a missing value in test dataset. We use median to replace it.

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# # Modeling and prediction

# In[ ]:


# Pearson Correlation of Features

# The plot indicts that there are not too many features strongly correlated with one another. 

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


# define training and testing sets

X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)


# In[ ]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


# Support Vector Machines

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


# KNN

knn = KNeighborsClassifier(n_neighbors = 3)
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


# Model evaluation
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Random Forest','KNN', 'Naive Bayes'],
    'Score': [ logreg.score(X_train, Y_train),svc.score(X_train, Y_train), random_forest.score(X_train, Y_train), knn.score(X_train, Y_train), gaussian.score(X_train, Y_train)]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


# Random Forest model feature importance 

importances = random_forest.feature_importances_
indices = np.argsort(importances)
features = X_train.columns
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic_prediction.csv', index=False)

