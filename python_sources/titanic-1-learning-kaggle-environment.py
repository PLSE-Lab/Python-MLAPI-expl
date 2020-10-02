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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import model_selection

# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Sometimes it makes more sense to work with the full dataset
full_data = [train,test]

# SUMMARIZE THE DATA
# Look at the info for the DataFrame 
print(train.info())

# Take a look at some rows in the datasets
# Training set
print(train.head(5))

#Test set
# print(test.head(5))

# Describe the datasets
# Training set

print(train.describe())

# Test set
print(test.describe())

# ENHANCE THE DATA

# Create a new column for total family size. Use full_data
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

# Create a new column to identify people travelling alone
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

#Create a new column to hold a person's title
for dataset in full_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# View Titles to Pclass
# print(pd.crosstab(train['Title'], train['Pclass']))
# print(pd.crosstab(test['Title'], test['Pclass']))

#Replace foreign titles for English equivalents. Lady and sir used where they are all first class passengers
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Lady')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Lady')
    dataset['Title'] = dataset['Title'].replace('Don', 'Sir')
    dataset['Title'] = dataset['Title'].replace('Dona', 'Lady')
    dataset['Title'] = dataset['Title'].replace('Jonkheer', 'Sir')

# Check that people's titles match their sex
# print(pd.crosstab(train['Title'], train['Sex']))
# print(pd.crosstab(test['Title'], test['Sex']))

# CONVERT CATEGORIES TO NUMBERS
# Titles to numbers
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Capt": 5, "Col": 6, "Countess": 7, "Dr": 8, "Lady": 9, "Major": 10, "Rev": 11, "Sir": 12}
for dataset in full_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

for dataset in full_data:
    dataset.Fare = dataset.Fare.round()
    

# ESTIMATE BLANK VALUES
# Create an empty matrix for age estimates
guess_ages = np.zeros((2,3))
print(guess_ages)

# Find median ages based on Sex and Pclass 
for dataset in full_data:
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
            
# Iterate over Sex and Pclass. Fill in null entries for age 

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

# Estimate the missing ports of embarkation. There are only two so use the most frequent value
freq_port = train.Embarked.dropna().mode()[0]
print(freq_port)
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# Ports of embarkation to numbers
ports_mapping = {"S": 1, "C": 2, "Q": 3}
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].map(ports_mapping)
    dataset['Embarked'] = dataset['Embarked'].fillna(0)

# Estimate the one missing fare in test
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

# Drop Name
train = train.drop(["Name"], axis=1)
test = test.drop(["Name"], axis=1)
# Drop cabin
train = train.drop(["Cabin"], axis=1)
test = test.drop(["Cabin"], axis=1)
#Drop ticket
train = train.drop(["Ticket"], axis=1)
test = test.drop(["Ticket"], axis=1)
#Drop SibSp
train = train.drop(["SibSp"], axis=1)
test = test.drop(["SibSp"], axis=1)
#Drop Parch
train = train.drop(["Parch"], axis=1)
test = test.drop(["Parch"], axis=1)

#Drop passengerId for train only
train = train.drop(["PassengerId"], axis=1)



# Print info on our datasets to see what work remains to complete fields
print(train.info())
print(test.info())
print(train.head())
print(test.head())

# Split-out validation dataset
X = train.drop("Survived", axis=1)
Y = train["Survived"]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn
results = []
mnames = []
for mname, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	mnames.append(mname)
	msg = "%s: %f (%f)" % (mname, cv_results.mean(), cv_results.std())
	print(msg)

#X_train = train.drop("Survived", axis=1)
#Y_train = train["Survived"]
#X_test  = test.drop("PassengerId", axis=1).copy()
#print(X_train.shape, Y_train.shape, X_test.shape)	

# Make predictions on validation dataset
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
predictions = rf.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

