# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Changing the current working directory.

#setcwd("../input/")

# Reading the file "train.csv" into a dataframe.
# This file contains all the data including the column 'Survived', using this we need to train our algorithm.

train=pd.read_csv("../input/train.csv")

# Reading the file "test.csv" into a dataframe.
# In this file we need to predict the column 'Survived' using our algorithm(Survived=1, not_survived=0).

test=pd.read_csv("../input/test.csv")

# Examining and cleaning the data in "train.csv".

print(train.describe())

# The 'Age' column has only 714 entries whereas other columns have 819 entires.
# The missing values in the column 'Age' are filled by the median value.

train["Age"]=train["Age"].fillna(train["Age"].median())

# Only numeric columns can be used in machine learning algorithm so we need to convert non-numeric columns to numeric values.
# The first column which comes to interest is 'Sex' because the titanic incidcent was famous for "women and children first".
# The unique values in the 'Sex' column.

print(train["Sex"].unique())

# The 'Sex' column has two values 'male' and 'female'.
# Assigning 'male'=0, 'female'=1(converting into numeric values).

train.loc[train["Sex"]=="male","Sex"]=0
train.loc[train["Sex"]=="female","Sex"]=1

# The second thing which can help us is 'Embarked' column, because the position of cabin of passengers may depend on the location of boarding.
# The unique values in the 'Embarked' column.

print(train["Embarked"].unique())

# The 'Embarked' column has four values 'S', 'Q', 'C', 'nan' in which 'nan' represents empty values.
# Finding out the ferquency of the values.

s=pd.Series(train["Embarked"])
s.value_counts()

# The 'S' value has the highest frequency, so the empty values are filled with the value of 'S'.

train["Embarked"]=train["Embarked"].fillna("S")

# Assigning 'S'=0, 'C'=1, 'Q'=2(converting into numeric values).

train.loc[train["Embarked"]=="S","Embarked"]=0
train.loc[train["Embarked"]=="C","Embarked"]=1
train.loc[train["Embarked"]=="Q","Embarked"]=2

# The next thing to be considered is "large families will take more time to get together and escape".
# For this a new variable 'family_size' is created which is sum of 'SibSp' and 'Parch'.
# First copying the data in train to train_copy.

train_copy=train.copy()

# Creating the 'family_size' variable.

train_copy["family_size"]=train_copy["SibSp"]+train_copy["Parch"]+1

# Now the data is set for analysis.



# Creating the model.

target=train_copy["Survived"].values
model_features=train_copy[["Pclass","Age","Sex","Fare","SibSp","Parch","Embarked","family_size"]].values

# Importing the necessary packages.

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import ensemble
from sklearn import cross_validation


#using voting classifier

rf = RandomForestClassifier(n_estimators=500,max_features='sqrt',bootstrap=True,n_jobs=2,class_weight={0:.2, 1:.8}) 
lr = LogisticRegressionCV(penalty='l2', class_weight={0:.2, 1:.8},cv=3,n_jobs=-1,refit=True)
nb = GaussianNB()
nb.class_prior_ = [0.6, 0.4]
gb = GradientBoostingClassifier(loss='deviance',learning_rate=0.001,n_estimators=500,max_depth=5,subsample=0.7)

my_model = ensemble.VotingClassifier(estimators=[('rf', rf), ('lr',lr), ('nb', nb), ('gb', gb)], voting='soft')
my_model = my_model.fit(model_features,target) 

# Before using the above model, we need to clean the data in "test.csv" file.

print(test.describe())

# Cleaning the data in the file "test.csv".
# The columns 'Fare' and 'Age' have missing values and these are filled with their median values respectively.

test["Age"]=test["Age"].fillna(test["Age"].median())
test["Fare"]=test["Fare"].fillna(test["Fare"].median())

# Changing the non-numeric values to numeric values in the file "test.csv".

test.loc[test["Embarked"]=="S","Embarked"]=0
test.loc[test["Embarked"]=="C","Embarked"]=1
test.loc[test["Embarked"]=="Q","Embarked"]=2
test.loc[test["Sex"]=="male","Sex"]=0
test.loc[test["Sex"]=="female","Sex"]=1

# Creating the family_size variable in the file "test.csv"

test_copy=test.copy()
test_copy["family_size"]=test_copy["SibSp"]+test_copy["Parch"]+1

# Making the prediction.

test_features=test_copy[["Pclass","Age","Sex","Fare","SibSp","Parch","Embarked","family_size"]].values
predictions=my_model.predict(test_features)

# Making the submission.

submission=pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":predictions})
submission.to_csv("submission2.csv",index=False)













