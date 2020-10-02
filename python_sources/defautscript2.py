# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

print("Hello")

print(train_df.columns.values)

# preview the data - This is the first 5 columns.
print(train_df.head())
# This is the last 5 columns.
print(train_df.tail())

# Gives the column data types.
train_df.info()
print('_'*40)
test_df.info()

print()

# Gives statistical information on all numeric fields.
print(train_df.describe())

print()

# Gives information about non-numeric fields.
print(train_df.describe(include=['O']))

print()

#Groups by class and gives a percentage of who survived.
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#Groups by sex.
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#Groups by Sibling/Spouse.
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#Groups by Parent/Child.
print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print()

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

print()

#Cross reference title with sex.
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))

print()

#Breaks down different titles into classifications and gives percentage that survive.
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

#Convert titles into ordinals for easier handling.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

#Drop the Name and Passenger ID columns from the training set.
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

#Convert Sex into 0 (female) or 1 (male).
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()

#Guess ages.
guess_ages = np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
print()
print(train_df.head())

#Create age bands for survival correlation.
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()

#Create family size feature.
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Create is alone feature.
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

#Drop the Parent/Child, Sibling, and Family Size features for isAlone.
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]
print()
print(train_df.head())

#Create artificial feature combining Age and Class.
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

#Determine survival based on port of embarkation.
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

print()    
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#Convert the embarkation point to numerics.
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()

combine = [train_df, test_df]
    
print()
print("The Training Set")
print(train_df.head(10))
print()
print("And... The Test Set")
print(test_df.head(10))

#Build the training and test sets.
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

print()
print (X_train)
print (Y_train)
print (X_test)
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
#logreg.predict(X_test)
#acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
#print(acc_log)

#Support Vector Machine
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

print("**********************************")


import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC

#Print you can execute arbitrary python code
train_data = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_data = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train_data.head())

print("\n\nSummary statistics of training data")
print(train_data.describe())

print("\n\nTraining number")
train_num = train_data.shape[0]
print(train_num)

#Any files you save will be available in the output tab below
train_data.to_csv('copy_of_the_training_data.csv', index=False)

full_data = train_data.append(test_data, ignore_index = True)
titanic_data = full_data[:train_num]

# prepare feature
from sklearn.preprocessing import LabelEncoder

# sex and embarked feature
sex = pd.Series( np.where( full_data.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
embarked = pd.get_dummies( full_data.Embarked , prefix='Embarked' )
le = LabelEncoder()

# age and fare feature
imputed = pd.DataFrame()
imputed[ 'Age' ] = full_data.Age.fillna( full_data.Age.mean() )
imputed[ 'Fare' ] = full_data.Fare.fillna( full_data.Fare.mean() )
imputed[ 'Parch' ] = full_data.Parch.fillna(full_data.Parch.mean())
imputed[ 'SibSp' ] = full_data.SibSp.fillna(full_data.SibSp.mean())

 
# title feature
title = pd.DataFrame()
title['Title'] = full_data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip() )

Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Dr":         "Officer",
                    "Rev":        "Officer",                    
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Lady":       "Royalty",                    
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr":         "Mr",
                    "Mrs":        "Mrs",
                    "Miss":       "Miss",
                    "Master":     "Master",
                    }

title['Title'] = title.Title.map(Title_Dictionary)
title = pd.get_dummies(title.Title)

# cabin feature
cabin = pd.DataFrame()
cabin['Cabin'] = full_data.Cabin.fillna('U')
cabin['Cabin'] = cabin.Cabin.map(lambda c: c[0])
cabin = pd.get_dummies(cabin.Cabin, prefix = 'Cabin')
cabin.head()

from sklearn.tree import DecisionTreeClassifier


