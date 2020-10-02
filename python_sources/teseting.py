import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
target = train["Survived"].values
copyOfTrain=train
copyOfTest=test


#Convert the male and female groups to integer form
copyOfTrain["Sex"][train["Sex"] == "male"] = 0
copyOfTrain["Sex"][train["Sex"] == "female"] = 1
#Impute the Embarked variable
copyOfTrain["Embarked"]=train["Embarked"].fillna("S")

#Convert the Embarked classes to integer form
copyOfTrain["Embarked"][train["Embarked"] == "S"] = 0
copyOfTrain["Embarked"][train["Embarked"] == "C"] = 1
copyOfTrain["Embarked"][train["Embarked"] == "Q"] = 2

print(len(copyOfTrain))
# Impute the missing value with the median
copyOfTest.Fare[152] = test["Fare"].median()
copyOfTrain = copyOfTrain.apply(lambda x:x.fillna(x.value_counts().index[0]))

#Convert the male and female groups to integer form
copyOfTest["Sex"][train["Sex"] == "male"] = 0
copyOfTest["Sex"][train["Sex"] == "female"] = 1
#Impute the Embarked variable
copyOfTest["Embarked"]=train["Embarked"].fillna("S")

#Convert the Embarked classes to integer form
copyOfTest["Embarked"][train["Embarked"] == "S"] = 0
copyOfTest["Embarked"][train["Embarked"] == "C"] = 1
copyOfTest["Embarked"][train["Embarked"] == "Q"] = 2

copyOfTest = copyOfTrain.apply(lambda x:x.fillna(x.value_counts().index[0]))
# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = copyOfTrain[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest,target)
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = copyOfTest[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(copyOfTest["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])

print(len(copyOfTest))

#Any files you save will be available in the output tab below
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])
print(features_forest)
print(target)