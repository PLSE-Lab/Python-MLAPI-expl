# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
pd.options.mode.chained_assignment = None
# Any results you write to the current directory are saved as output.

train_titanic = pd.read_csv("../input/train.csv")
test_set = pd.read_csv("../input/test.csv")
# dataset = train_titanic
def treat(dataset):
    dataset["FirstName"] = dataset.Name.str.split(",")
    dataset["LastName"] = dataset.FirstName.str[0]
    dataset.FirstName = dataset.FirstName.str[1]
    dataset["Title"] = dataset.FirstName.str.split(".")
    dataset.FirstName = dataset.Title.str[1]
    dataset["Title"] = dataset.Title.str[0]
    dataset.Title = dataset.Title.str.strip(" ")
    
    
    dataset.Embarked[dataset.Embarked == "C"] = 1 # first port
    dataset.Embarked[dataset.Embarked == "Q"] = 2 # second port
    dataset.Embarked[dataset.Embarked == "S"] = 3 # last port
    dataset.Embarked = dataset.Embarked.fillna(dataset.Embarked.mode().iloc[0]) # mode imputation for the port of entry
    
    # making a vector of priority based on the tile worst = 0 best = 5
    dataset.Title = dataset.Title.replace(["Col", "Capt", "Major"], 1) # assuming crew and military are the last ones to leave the ship
    dataset.Title = dataset.Title.replace(["Mr", "Sir", "Jonkheer"], 2) 
    dataset.Title = dataset.Title.replace(["Rev", "Don", "Dr"], 3) # Doctors and priests have an important social standing in society so they are given some priority
    dataset.Title = dataset.Title.replace(["Miss", "Mlle", "Ms"], 4) 
    dataset.Title = dataset.Title.replace(["Mrs", "Mme", "Master", "Dona"], 5) # master is always children assuming that children and married woman have the second highest priority
    dataset.Title = dataset.Title.replace(["Lady", "the Countess"], 6) # assuming noble women have the highest prority
    
    # Inputation for "nan" values in Age
    dataset.Age = SimpleImputer().fit_transform(pd.DataFrame(dataset.Age))
    # NaN_age = np.array(dataset.Age[np.isnan(dataset.Age)].index.to_series())
    # who_NaN_age = dataset.iloc[NaN_age]
    dataset.Sex = dataset.Sex.replace(["male"], 0) # I'm not sure whether specifying the gender is necessary if the priority is already given by the title
    dataset.Sex = dataset.Sex.replace(["female"], 1)

    #dataset["SibSp2"] = # nothing = 0 , sib = 1, spouse = 2
    #dataset.SibSp = 

    return dataset

train_titanic_treated = treat(train_titanic)
test_set_treated = treat(test_set)

features = ["Pclass","Title", "Sex", "Age", "SibSp", "Parch","Embarked"]

y = train_titanic_treated.Survived
X = train_titanic_treated[features]

titanic_model = DecisionTreeClassifier(random_state=0)

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
titanic_model.fit(train_X, train_y)

val_predictions = titanic_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)
print("Validate predictions\n",val_mae)

X_test = test_set_treated[features]
y_test = titanic_model.predict(X_test)


# stole this from Leon :)

pID = list(test_set_treated['PassengerId'])
final = pd.DataFrame(columns=['PassengerId', 'Survived'])
final['Survived'], final['PassengerId'] = y_test, pID
final.to_csv(path_or_buf="franco_prediction.csv", index=False)
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(test_set_treated.describe())