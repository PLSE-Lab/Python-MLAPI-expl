import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Convert Sex into workable feature, 1 for female and 0 for male passenger
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})
# Impute missing age values with median age
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())
# New feature of below age
train['Minor?'] = (train['Age']<15)
train['Minor?'] = train['Minor?'].astype(int)
test['Minor?'] = (test['Age']<15)
test['Minor?'] = test['Minor?'].astype(int)
#View NaNs
#print(train[train.isnull().any(axis=1)])
#print(test[test.isnull().any(axis=1)])
# Too many NaN cabins to be of value
del train['Cabin']
del test['Cabin']
# Impute 3 NaN rows overall, should not affect too much
test['Fare'] = test['Fare'].fillna(test['Fare'].median())


# Select relevant features
#trainX = train.select_dtypes(exclude=['object'])
#testX = test.select_dtypes(exclude=['object'])
trainX = train[['Survived','Pclass','Minor?','Sex']]
testX = test[['Pclass','Minor?','Sex']]
# Select output variable
trainY = trainX['Survived'].copy()
del trainX['Survived']

# Flatten into a 1-D array
trainY = np.ravel(trainY)

# Create multilayer perceptron model
clf = DecisionTreeClassifier()
clf.fit(trainX,trainY)
print(trainX.columns)
print(clf.feature_importances_)

# Create prediction file
predictions = clf.predict(testX)
submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId'].copy()
submission['Survived'] = predictions

#Any files you save will be available in the output tab below
submission.to_csv('submission.csv', index=False)