import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
y=[]
for x in range(0, 3):
    y.append(x)
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
target = train["Survived"].values


copyOfTrain=train
copyOfTest=test

#Convert the male and female groups to integer form
copyOfTrain["Sex"][train["Sex"] == "male"] = 0
copyOfTrain["Sex"][train["Sex"] == "female"] = 1

copyOfTest["Sex"][test["Sex"] == "male"] = 0
copyOfTest["Sex"][test["Sex"] == "female"] = 1

#Impute the Embarked variable
copyOfTrain["Embarked"]=train["Embarked"].fillna("S")

copyOfTest["Embarked"]=test["Embarked"].fillna("S")

#Convert the Embarked classes to integer form
copyOfTrain["Embarked"][train["Embarked"] == "S"] = 0
copyOfTrain["Embarked"][train["Embarked"] == "C"] = 1
copyOfTrain["Embarked"][train["Embarked"] == "Q"] = 2

copyOfTest["Embarked"][test["Embarked"] == "S"] = 0
copyOfTest["Embarked"][test["Embarked"] == "C"] = 1
copyOfTest["Embarked"][test["Embarked"] == "Q"] = 2


# Impute the missing value with the median
copyOfTest.Fare[152] = test["Fare"].median()
copyOfTrain = copyOfTrain.apply(lambda x:x.fillna(x.value_counts().index[0]))
copyOfTest = copyOfTest.apply(lambda x:x.fillna(x.value_counts().index[0]))


Train2=copyOfTrain[0:600] 
print(Train2.shape)
CV2=copyOfTrain[601:891]
print(CV2.shape)
Target2 = Train2["Survived"].values
ground_truth=CV2["Survived"].values




# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = Train2[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

#final
# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = copyOfTrain[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(max_depth=10, min_samples_split=10, n_estimators = 138, random_state = 1)
my_forest = forest.fit(features_forest,target)
# Compute predictions on our test set features then print the length of the prediction vector
test_features = copyOfTest[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])


#Any files you save will be available in the output tab below
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])






