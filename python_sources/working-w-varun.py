import numpy as np
import pandas as pd
import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# Drop columns we don't care about
train = train.drop(['Name', 'Cabin', 'Ticket'], 1)

# Convert non-quantitative data to quantitative data
labels = {"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}}
train["Sex"] = train["Sex"].replace(labels["Sex"])
train["Embarked"] = train["Embarked"].replace(labels["Embarked"])
# Fill nan values
train = train.fillna(3)

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

# Random Forest Classifier
estimator = RandomForestClassifier(random_state=0, n_estimators=100)
score = cross_val_score(estimator, train.drop('Survived', 1), train['Survived']).mean()
print("\n\nCross Validation Score of Random Forests")
print(str(score))

##  Predict against test set
test = test.drop(['Name', 'Cabin', 'Ticket'], 1)
# Convert non-quantitative data to quantitative data
labels = {"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}}
test["Sex"] = test["Sex"].replace(labels["Sex"])
test["Embarked"] = test["Embarked"].replace(labels["Embarked"])
# Fill nan values
test = test.fillna(3)

estimator.fit(train.drop('Survived', 1), train['Survived'])
predictions = estimator.predict(test).astype(int)

# Print predictions
print("\n\nTop of the predictions:")
print(predictions[:10])

ids = pd.to_numeric(test['PassengerId'])

# Writing predictions to a file
predictions_file = open("myfirstforest.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, predictions))
predictions_file.close()