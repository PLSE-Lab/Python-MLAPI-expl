import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
from matplotlib.pyplot import hist
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score
import os
print(os.listdir("../input"))

# input
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_gender_submission = pd.read_csv('../input/gender_submission.csv')
# columns
df_train.columns.values
df_test.columns.values
# nan values
df_train.info()
sns.heatmap(df_train.isna())
df_test.info()
sns.heatmap(df_test.isna())
# describe
pd.set_option('display.max_columns', None)
df_train.describe()
df_test.describe()
# head
df_train.head()
df_test.tail()


# LabelEncoder 'Sex'
le1 = preprocessing.LabelEncoder()
le1.fit(df_train['Sex'])
df_train['Sex_1'] = le1.transform(df_train['Sex'])
keys1 = le1.classes_
values1 = le1.transform(le1.classes_)
dictionary1 = dict(zip(keys1, values1))
print(dictionary1)

sns.heatmap(df_train.corr(), annot = True)

# Survived
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False).plot()
df_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# graphs
sns.catplot(x="Sex", hue="Pclass", col="Survived",data=df_train, kind="count",height=4, aspect=.7);
sns.catplot(x="SibSp", hue="Pclass", col="Survived",data=df_train, kind="count",height=4, aspect=.7);
sns.catplot(x="Parch", hue="Pclass", col="Survived",data=df_train, kind="count",height=4, aspect=.7);

g = sns.FacetGrid(df_train, col='Survived', row='Pclass')
g = g.map(plt.hist, "Sex_1")
g.add_legend()

g = sns.FacetGrid(df_train, col='Survived', row='Pclass')
g = g.map(plt.hist, "Age")
g.add_legend()

g = sns.FacetGrid(df_train, row='Embarked')
g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
g.add_legend()

df_train[df_train['Survived']== 1]['Age'].hist()
df_train[df_train['Survived']== 0]['Age'].hist()

#regex
df_train['Title'] = df_train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df_test['Title'] = df_test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(df_train['Title'], df_train['Sex'])
# rename - replace
df_train['Title'] = df_train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df_train['Title'] = df_train['Title'].replace('Mlle', 'Miss')
df_train['Title'] = df_train['Title'].replace('Ms', 'Miss')
df_train['Title'] = df_train['Title'].replace('Mme', 'Mrs')
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

df_test['Title'] = df_test['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df_test['Title'] = df_test['Title'].replace('Mlle', 'Miss')
df_test['Title'] = df_test['Title'].replace('Ms', 'Miss')
df_test['Title'] = df_test['Title'].replace('Mme', 'Mrs')

# LabelEncoder 'Title'
le2 = preprocessing.LabelEncoder()
le2.fit(df_train['Title'])
df_train['Title_1'] = le2.transform(df_train['Title'])
keys2 = le2.classes_
values2 = le2.transform(le2.classes_)
dictionary2 = dict(zip(keys2, values2))
print(dictionary2)

# groups of Ages
bins = [0, 20, 30, 40, 50, 60, 70, 80, np.inf]
names = ['<20', '20-30', '30-40', '40-50', '50-60', '60-70','70-80' ,'90+']
df_train['AgeRange'] = pd.cut(df_train['Age'], bins, labels=names)

bins = [0, 20, 30, 40, 50, 60, 70, 80, np.inf]
names = ['<20', '20-30', '30-40', '40-50', '50-60', '60-70','70-80' ,'90+']
df_test['AgeRange'] = pd.cut(df_test['Age'], bins, labels=names)

df_train[['AgeRange', 'Survived']].groupby(['AgeRange']).count().sort_values(by='AgeRange', ascending=True).plot()

#stats AgeRange
sns.catplot(x="Pclass", hue='AgeRange', col="Survived",data=df_train, kind="count",height=4, aspect=.7);
sns.catplot(x="Pclass", hue='AgeRange', col="Sex",data=df_train, kind="count",height=4, aspect=.7);
sns.catplot(x="Pclass", hue='AgeRange', col="Sex",data=df_train, kind="count",height=4, aspect=.7);
sns.catplot(x="AgeRange", y="Fare", kind="box", data=df_train);
sns.catplot(x="Pclass", y="Fare", kind="box", data=df_train);


# logistic regression
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

X_train = df_train.drop(["PassengerId", "Survived",'AgeRange', 'Name', 'Sex', 'Sex_1','Ticket', 'Cabin', 'Embarked', 'Sex','Title', 'Title_1'], axis=1)
Y_train = df_train["Survived"]
X_test  = df_test.drop(["PassengerId", 'AgeRange', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the pred
acc_log = round(classifier.score(X_train, Y_train) * 100, 2)
acc_log