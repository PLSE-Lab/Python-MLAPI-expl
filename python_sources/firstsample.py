import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.ensemble import RandomForestClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

le = LabelEncoder()
train['sex'] = le.fit_transform(train['Sex'].values)
test['sex'] = le.transform(test['Sex'].values)

imr1 = Imputer(missing_values='NaN', strategy='median', axis=0).fit(train[['Age']])
imr2 = Imputer(missing_values='NaN', strategy='median', axis=0).fit(train[['Fare']])

train['age'] = imr1.transform(train[['Age']].values)
test['age'] = imr1.transform(test[['Age']].values)

train['fare'] = imr2.transform(train[['Fare']].values)
test['fare'] = imr2.transform(test[['Fare']].values)

X_train = train[['Pclass', 'sex', 'age', 'fare']].values
X_test = test[['Pclass', 'sex', 'age', 'fare']].values
y_train = train['Survived'].values

rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_train, y_train)

test['Survived'] = rf.predict(X_test)


#Any files you save will be available in the output tab below
test[['PassengerId', 'Survived']].to_csv('myoutput.csv', index=False)

print(test.describe())
print(train.describe())