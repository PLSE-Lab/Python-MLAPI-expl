# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv('../input/train.csv')
test_file_data = pd.read_csv('../input/test.csv')
#print(data)

data = data.drop(columns=['Ticket', 'Cabin', 'Name'])
data = data.dropna()
test_file_data = test_file_data.drop(columns=['Ticket', 'Cabin', 'Name'])
test_file_data = test_file_data.fillna(test_file_data['Age'].mean())

le = preprocessing.LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
test_file_data['Sex'] = le.fit_transform(test_file_data['Sex'])

le = preprocessing.LabelEncoder()
data['Embarked'] = le.fit_transform(data['Embarked'].astype(str))
test_file_data['Embarked'] = le.fit_transform(test_file_data['Embarked'].astype(str))

train_data = data[0:600]
test_data = data[600:891]

train_labels = train_data.Survived
test_labels = test_data.Survived

train_data = train_data.drop(columns=['Survived'])
test_data = test_data.drop(columns=['Survived'])

print(data)

clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)
clf = clf.fit(train_data, train_labels)
test_predictions = clf.predict(test_data)


print(classification_report(test_labels, test_predictions)) 
print("Accuracy: ", accuracy_score(test_labels, test_predictions))
print("Confusion Matrix: ") 
print(confusion_matrix(test_labels, test_predictions))

test_file_predictions = clf.predict(test_file_data)

submission = pd.DataFrame({ 'PassengerId': test_file_data['PassengerId'],
                            'Survived': test_file_predictions })
submission.to_csv("submission.csv", index=False)

# Any results you write to the current directory are saved as output.