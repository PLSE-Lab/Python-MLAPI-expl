# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Read the input datasets
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Fill missing numeric values with median for that column
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)



# Encode sex as int 0=female, 1=male
train_data['Sex'] = train_data['Sex'].apply(lambda x: int(x == 'male'))


# Extract pclass, sex, age, fare features
X = train_data[['Pclass', 'Sex', 'Age', 'Fare']].as_matrix()
print(np.shape(X))

# Extract survival target
y = train_data[['Survived']].values.ravel()
print(np.shape(y))


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
model =RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=4, min_samples_split=3, min_samples_leaf=2)
#model= KNeighborsClassifier(3)
#model=SVC(probability=True)
#model=DecisionTreeClassifier()
#model=AdaBoostClassifier()
#model=GradientBoostingClassifier()
# Create model with all training data
classifier = model.fit(X, y)

# Create model with all training data
classifier = model.fit(X, y)

# Encode sex as int 0=female, 1=male
test_data['Sex'] = test_data['Sex'].apply(lambda x: int(x == 'male'))

# Extract pclass, sex, age, fare features
X_ = test_data[['Pclass', 'Sex', 'Age', 'Fare']].as_matrix()

# Predict if passengers survived using model
y_ = classifier.predict(X_)

# Append the survived attribute to the test data
test_data['Survived'] = y_
predictions = test_data[['PassengerId', 'Survived']]
print(predictions)

# Save the output for submission
predictions.to_csv('submission.csv', index=False)