# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import *
from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train_file = '../input/train.csv'
test_file = '../input/test.csv'
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
all_data = pd.concat((train.loc[:,'Pclass':'Embarked'], test.loc[:,'Pclass':'Embarked']))
all_data = all_data.drop(columns = ['Name', 'Ticket', 'Cabin'])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
scaler = MinMaxScaler()
X_train = scaler.fit_transform(all_data[:train.shape[0]])
X_test = scaler.transform(all_data[train.shape[0]:])
y = train.Survived
model = GradientBoostingClassifier(
    criterion="mse", n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0
).fit(X_train, y)
preds = model.predict(X_test)
solution = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": preds})
solution.to_csv("solution.csv", index = False)
