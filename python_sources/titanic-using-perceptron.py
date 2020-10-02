import pandas as pd
import numpy as np
import sklearn.model_selection as ms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv("../input/titanic/train.csv")
Y = np.array(data["Survived"])
X = data.drop(['Survived', 'PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)

# Standardization of Values
X['Sex'] = X['Sex'].replace({"male": 1, "female": 0})
X['Fare'] = (X['Fare'] - min(X['Fare'])) / (max(X['Fare']) - min(X['Fare']))
X['Pclass'] = (X["Pclass"] - min(X['Pclass'])) / (max(X['Pclass']) - min(X['Pclass']))
X = X.to_numpy()

X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)


class perceptron:
    def __init__(self):
        self.w = None
        self.b = None

    def model(self, x):
        return 1 if (np.dot(self.w, x) >= self.b) else 0

    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)

        return np.array(Y)

    def fit(self, X, Y,epochs=1):
        self.w = np.ones(X.shape[1])
        self.b = 0
        accuracy=np.array(range(epochs))
        for i in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + x
                    self.b = self.b + 1
                elif y == 0 and y_pred == 1:
                    self.w = self.w - x
                    self.b = self.b - 1

            accuracy[i]= accuracy_score(self.predict(X),Y)


perceptron = perceptron()
perceptron.fit(X_train,Y_train,10)

testset=pd.read_csv("../input/titanic/test.csv")

testset = testset.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)
# Standardization of Values
testset['Sex'] = testset['Sex'].replace({"male": 1, "female": 0})
testset['Fare'] = (testset['Fare'] - min(testset['Fare'])) / (max(testset['Fare']) - min(testset['Fare']))
testset['Pclass'] = (testset["Pclass"] - min(testset['Pclass'])) / (max(testset['Pclass']) - min(testset['Pclass']))
testset = testset.to_numpy()

y_pred_test=perceptron.predict(testset)

finalsubmission=pd.read_csv("../input/titanic/gender_submission.csv")
final=finalsubmission.drop(['Survived'],axis=1)
final["Survived"]=y_pred_test
print(final)
final.to_csv("final_submission.csv",index=False)