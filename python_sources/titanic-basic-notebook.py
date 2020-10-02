# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

file_path = "../input/titanic/train.csv"
data = pd.read_csv(file_path) 
data.Sex = data.Sex.map({'male':1, 'female':0})
data.Age = data.Age.fillna(29.5)
y = data.Survived
features = ['Pclass','SibSp','Parch','Sex','Age']
X = data[features]
model = RandomForestRegressor(random_state=1)
model.fit(X, y)
test_path = "../input/titanic/test.csv"
test_data = pd.read_csv(test_path)
test_data.Sex = test_data.Sex.map({'male':1, 'female':0})
test_data.Age = test_data.Age.fillna(29.5)
test_X = test_data[features]
test_preds = model.predict(test_X).round().astype('int64')
output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Survived': test_preds})
output.to_csv('submission.csv', index=False)

# Any results you write to the current directory are saved as output.