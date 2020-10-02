# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#import the train dataset
train_csv = "../input/titanic/train.csv"
train_data = pd.read_csv(train_csv)
train_data.head()
train_data.columns

#specifying prediction target
Y = train_data.Survived

#Encode categorical data
train_le = LabelEncoder()
train_data['Sex'] = train_le.fit_transform(train_data['Sex'])

#create predictive features
feature = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = train_data[feature]

#build model
train_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
train_model.fit(X, Y)

#import the test dataset
test_csv = "../input/titanic/test.csv"
test_data = pd.read_csv(test_csv)
test_data.head()

#Encode test dataset categorical data
test_le = LabelEncoder()
test_data['Sex'] = test_le.fit_transform(test_data['Sex'])

#create test dataset predictive features and check for missing values
test_X = test_data[feature]
test_X.isnull().values.any()

#predict test file using the built model
y_pred = train_model.predict(test_X)
print(y_pred)

#create survived column by filling it prediction result
output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                      'Survived': y_pred})

#save output to csv file
output.to_csv('gender_submission.csv',index=False)