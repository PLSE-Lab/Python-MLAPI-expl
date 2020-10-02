# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#load training and testing dataframe from input
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

#check dataframe row/column names
train_df.index
train_df.columns
train_df.columns.values
#show top 10 rows, default is 5
train_df.head(10)
#show last 5 rows
train_df.tail()

#dataframe information
train_df.info()

#data features
#by default, only list nemeric fields
train_df.describe()
#or
train_df.describe(include=['number'])
#object fields
train_df.describe(include=['O'])
#or
train_df.describe(include=['object'])
#all fields
train_df.describe(include='all')

#check Survived in each Pclass
#as_index=False make result a DataFrame and assign row number starts from 0 as index
train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#check Survived in each Sex
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#check Survived in each Sibsp
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#check Survived in each Parch
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#check Survived in each Parch
train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#class+sex survival rate, if rich people more or less gentleman
train_df[['Pclass', 'Sex', 'Survived']].groupby(['Pclass', 'Sex']).mean()

#check aboard count in each Embarked port
train_df[['Embarked', 'Name']].groupby(['Embarked']).count()
#or use Series.value_counts()
train_df['Pclass'].value_counts()
#pclass and survived
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#check Sex count from each Embarked port
train_df[['Embarked', 'Sex', 'Name']].groupby(['Embarked', 'Sex']).count()
#check Pclass count from each Embarked port
train_df[['Embarked', 'Pclass', 'Name']].groupby(['Embarked', 'Pclass']).count()

#explor dataframe
#difference of iloc and loc, iloc use row number, loc use row index(not necessarily integer)
#first row, returns a pandas.Series
train_df.loc[0]
train_df.iloc[0]
#first row returns a one-row dataframe
train_df.loc[[0]]
train_df.iloc[[0]]
#first 5 rows
train_df.iloc[0:5]
#train_df.loc[0:5]
#Name column
name_series = train_df.loc[:, 'Name']
#or
name_series = train_df['Name']
#Name and Sex Columns
name_sex_df = train_df[['Name', 'Sex']]
#rows that miss Embarked
train_df.loc[train_df.Embarked.isnull()]
#or
train_df[train_df.Embarked.isnull()]
#rows that Survived
train_df.loc[train_df['Survived']==1]
#or
train_df[train_df.Survived==1]
#or
survived_df = train_df.query('Survived==1')
#First row name
train_df.loc[0, 'Name']
#Names that don't have Embarked
train_df.loc[train_df.Embarked.isnull(), 'Name']

#transform datafram, rotate row-column
train_df.T

#data visualization, better do it in notebook
#A histogram is a graphical representation of the distribution of numerical data.
#It is an estimate of the probability distribution of a continuous variable (quantitative variable)
#correlating numerical features
#col='Survived', grid has 1 row and 2 columns, with Survived=0 and 1,
#size: height of each facet in inchs, aspect:width ratio
g=sns.FacetGrid(train_df, col='Survived', size=3, aspect=1.5)
#col='Survived', row='Sex', grid has 2 rows and 2 coumns, row1: male-0, male-1, row2: female-0, female-1
#g=sns.FacetGrid(train_df, row='Sex', col='Survived', size=3, aspect=1.5)
#3 rows(Embarked), 2 columns(Sex), and different color for Survived=0/1
#g=sns.FacetGrid(train_df, row='Embarked', col='Sex', hue='Survived', size=3, aspect=1.5)
#Set aesthetic parameters, optional and not necessary here
sns.set(style="ticks", color_codes=True)
#plot histogram, bins is number of bars across all 'Age'
g.map(plt.hist, 'Age', bins=20)
#customized bins, for Age 0-80, each bin with width 10, color='b'-Blue, 'r'-Red, 'y'-Yellow, 'g'-Green
bins = np.arange(0,80,10)
g.map(plt.hist, 'Age', bins=bins, color='b')

#pointplot, show point estimates and confidence intervals using scatter plot glyphs.
#each facet has 2 lines, one for male, one for female
grid=sns.FacetGrid(train_df, row='Embarked', size=3, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', pallete='deep')
grid.add_legend()
#3 rows, 2 columns, each facet has 1 line
grid=sns.FacetGrid(train_df, row='Embarked', col='Sex', size=3, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', pallete='deep')
grid.add_legend()

#Categorical plot
#barplot, show point estimates and confidence intervals as rectangular bars.
grid = sns.FacetGrid(train_df, row="Embarked", col='Survived')
#ci=None, no confidence interval, alpha, darkness of bar
#grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5)

#Analyse Title
#extract title from name
train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#cross tabulation, row and column
pd.crosstab(train_df['Title'], train_df['Sex'])
#pd.crosstab(train_df['Sex'], train_df['Survived'])

#group titles
train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')
#count of each title
print (train_df['Title'].value_counts())
#survived(0/1) count of each title
pd.crosstab(train_df['Title'], train_df['Survived'])

#convert title to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
combine = [train_df, test_df]
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train_df.head()

#drop columns, axis=1 denotes column, default axis=0 denotes row
train_df = train_df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
train_df.head()
combine=[train_df, test_df]
print (train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

#convert categorical features to numerical
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1}).astype(int)
train_df.info()
train_df.head()

#Complete missing data, Age
guess_ages = np.zeros((2,3))
guess_ages
for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            #use median age of people with same sex and pclass value
            guess_df = dataset[(dataset['Sex']==i)&(dataset['Pclass']==j+1)]['Age'].dropna()
            guess_ages[i,j] = guess_df.median()       
    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[(dataset.Age.isnull())&(dataset.Sex==i)&(dataset.Pclass==j+1), 'Age'] = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)
print (guess_ages)            
train_df.head(10)

#band Age and determine correlation with Survived
#cut age to 5 bands, max age=80, so each band=16
for dataset in combine:
    dataset['AgeBand'] = pd.cut(dataset['Age'], 5)
    
#as_index=False makes AgeBand a column so that sort_values(by='AgeBand') works
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

#convert Age to ordinals
for dataset in combine:
    dataset.loc[(dataset['Age']<=16), 'Age'] = 0
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32), 'Age'] = 1
    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48), 'Age'] = 2
    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64), 'Age'] = 3
    dataset.loc[(dataset['Age']>64), 'Age'] = 4
train_df.head(10)

#drop AgeBand
train_df = train_df.drop(['AgeBand'], axis=1)
test_df = test_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

#Complete missing Embarked
#use port that most people board the ship
#list port and passenger count
train_df['Embarked'].value_counts()
#or
train_df[['Embarked', 'Survived']].groupby(['Embarked']).count()
#or, find mode
freq_port = train_df.Embarked.dropna().mode()
#replace missing Embarked with mode
train_df.loc[train_df['Embarked'].isnull(), 'Embarked']=train_df.Embarked.dropna().mode()[0]
#convert Embarked to ordinals
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'Q':1, 'C':2}).astype(int)

#create FamilySize based on SibSp and Parch
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
#check FamilySize frequcy
train_df['FamilySize'].value_counts()
#FamilySize Survived relations
train_df[['FamilySize', 'Survived']].groupby('FamilySize', as_index=False).mean().sort_values(by='Survived', ascending=False)

#consider make family size 1,2,3,4 and >=5, since >=5 has low frequency and low survival rate(except for 7)

#create isAlone, is this necessary?? compare without
#create isAlone based on FamilySize
for dataset in combine:
    dataset['isAlone'] = 0
    dataset.loc[dataset['FamilySize']==1, 'isAlone'] = 1
train_df[['isAlone', 'Survived']].groupby('isAlone', as_index=False).mean()

#drop SibSp, Parch
train_df = train_df.drop(['SibSp', 'Parch'], axis=1)
test_df = test_df.drop(['SibSp', 'Parch'], axis=1)
combine = [train_df, test_df]

#adjust Fare based on FamilySize, since Fare is family based
train_df['Fare'] = train_df['Fare']/train_df['FamilySize']
test_df['Fare'] = test_df['Fare']/test_df['FamilySize']

#Complete missing Fare in test_df
test_df.loc[(test_df.Fare.isnull())&(test_df.Pclass==3), 'Fare']
pf=test_df[['Pclass', 'Fare']].groupby('Pclass', as_index=False).median()
test_df.loc[(test_df.Fare.isnull())&(test_df.Pclass==3), 'Fare'] = pf.loc[pf['Pclass']==3, 'Fare'].median()

#convert Fare to FareBand
train_df['FareBand'] = pd.qcut(train_df['Fare'], 6)
train_df[['FareBand', 'Survived']].groupby('FareBand').mean()
#convert Fare to ordinal
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
#drop FareBand
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

#drop FamilySize
train_df = train_df.drop(['FamilySize'], axis=1)
test_df = test_df.drop(['FamilySize'], axis=1)
combine = [train_df, test_df]

# Model, Predict and Solve
x_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
x_test = test_df.drop('PassengerId', axis=1).copy()
x_train.shape, y_train.shape, x_test.shape

#random forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train)*100, 2)
acc_random_forest

#submission
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_pred
    })
submission.head(20)
#submission.to_csv('linden_Titanic_submission.csv', index=False)