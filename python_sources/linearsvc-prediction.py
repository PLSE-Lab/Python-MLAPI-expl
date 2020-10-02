import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')

train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(train['Age'].median())

test['Fare'] = test['Fare'].fillna(train['Fare'].median())

train.loc[ train['Embarked'] == "S", "Embarked"] = 0
train.loc[ train['Embarked'] == "C", "Embarked"] = 1
train.loc[ train['Embarked'] == "Q", "Embarked"] = 2

test.loc[ test['Embarked'] == "S", "Embarked"] = 0
test.loc[ test['Embarked'] == "C", "Embarked"] = 1
test.loc[ test['Embarked'] == "Q", "Embarked"] = 2

train.loc[ train['Sex'] == "male", "Sex"] = 0
train.loc[ train['Sex'] == "female", "Sex"] = 1

test.loc[ test['Sex'] == "male", "Sex"] = 0
test.loc[ test['Sex'] == "female", "Sex"] = 1

#print("\n\nSummary statistics of training data")
#print(train)
#print(test)
#exit()


#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X_train = train[predictors]
y_train = train['Survived']

#print(X_train)
#print(y_train)

X_test = test[predictors]
#y_test = test['Survived']

#train linear SVM
lsvm = LinearSVC()
lsvm.fit(X_train,y_train)
predictions = lsvm.predict(X_test)
#print(predictions)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
print(submission)
submission.to_csv("linearSVM_submission.csv", index=False)    



#
