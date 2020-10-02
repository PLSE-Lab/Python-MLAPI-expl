# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier 
 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
#train = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
test = pd.read_csv("../input/test.csv")
#test = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")


# Map Sex to 0/1
# train.loc[train['Sex'] == 'male'] = 0
# train.loc[train['Sex'] == 'female'] = 1
train.Sex.replace(['male', 'female'], [1, 0], inplace=True)

# test.loc[test['Sex'] == 'male'] = 0
# test.loc[test['Sex'] == 'female'] = 1
test.Sex.replace(['male', 'female'], [1, 0], inplace=True)

train.Age = train.Age.fillna(train.Age.describe().mean())
test.Age = test.Age.fillna(test.Age.describe().mean())
test.Fare = test.Fare.fillna(test.Fare.describe().mean())

train.Embarked = train.Embarked.fillna('S')
train.Embarked.replace(['C','Q','S'],[0,1,2],inplace = True)

test.Embarked.replace(['C','Q','S'],[0,1,2],inplace = True)

# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values
features_train = train[["Pclass", "Sex", "Age", "Fare","Embarked"]].values

# Fit your first decision tree: train_dec_tree
train_dec_tree = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
# train_dec_tree = tree.DecisionTreeClassifier 
train_dec_tree = train_dec_tree.fit(features_train, target)

# Extract the test features
features_test = test[["Pclass", "Sex", "Age", "Fare","Embarked"]].values

print(train_dec_tree.feature_importances_)

# Make the prediction using the test set
predictions = train_dec_tree.predict(features_test)


# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(predictions, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])