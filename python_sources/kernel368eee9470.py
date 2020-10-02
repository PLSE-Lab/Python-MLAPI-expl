# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

titanic_train = pd.read_csv('../input/cleantitanic/cleanTitanicTrain.csv')
titanic_test = pd.read_csv('../input/cleantitanic/cleanTitanicTest.csv')
predictor_names=["Pclass", "Sex", "Age", "SibSp", \
                "Parch", "Fare", "Embarked"]


trainPredictors = titanic_train[predictor_names]
target_name=['Survived']
train_target = titanic_train['Survived']
test_predictors= titanic_test[predictor_names]
test_PIDs=titanic_test['PassengerId']

my_random_forest = RandomForestClassifier(n_estimators = 100, min_samples_split = 10, min_samples_leaf = 5, max_features = 2, oob_score = True)
my_random_forest.fit(X = titanic_train[predictor_names], y = titanic_train["Survived"])

print("OOB accuracy: ")
print(my_random_forest.oob_score_)

for feature, imp in zip(predictor_names, my_random_forest.feature_importances_):
    print(feature, imp)
    
test_predictions = my_random_forest.predict(X= titanic_test[predictor_names])
submission = pd.DataFrame({"PassengerId":titanic_test["PassengerId"], "Survived":test_predictions})
submission.to_csv("tutorial_randomForest_submission.csv", index = False)

# Any results you write to the current directory are saved as output.