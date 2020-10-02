import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X= pd.read_csv('../input/titanic/train.csv')
y=X.pop('Survived')

numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
X[numeric_variables].head()

X["Age"].fillna(X.Age.mean(), inplace=True)
X.tail()

model= RandomForestClassifier(n_estimators=1000)
model.fit(X[numeric_variables], y)

print("Train_acc:", accuracy_score(y, model.predict(X[numeric_variables])))

test =pd.read_csv("../input/titanic/test.csv")
test["Age"].fillna(test.Age.mean(), inplace=True)
test=test[numeric_variables].fillna(test.mean()).copy()

y_pred = model.predict(test[numeric_variables])


submission= pd.DataFrame({"PassengerId":test["PassengerId"],"Survived": y_pred})
submission.to_csv('submit.csv',index=False)

submission.head()


