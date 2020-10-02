import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# Replace all the occurences of male with the number 0
train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#Drop all other features
train = train.drop(['Parch', 'SibSp','Ticket', 'Cabin','Name', 'PassengerId','Pclass','Age','Fare','Cabin','Embarked'], axis=1)
test = test.drop(['Parch', 'SibSp','Ticket', 'Cabin','Name', 'Pclass','Age','Fare','Cabin','Embarked'], axis=1)

#Creating variable for regression
X_train = train.drop(["Survived"], axis=1)
Y_train = train["Survived"]
X_test  = test.drop(["PassengerId"], axis=1).copy()

#Implementing logical regression
logreg = LogisticRegression()
result = logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
predicted_probs = logreg.predict_proba(X_test)
#print(logreg.classes_)
print(acc_log)
print(logreg.coef_)


#submitting assignment
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)	

