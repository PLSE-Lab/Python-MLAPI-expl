# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
target = 'Survived'
passenger_id = 'PassengerId'

for dataset in [df_train, df_test]:
    dataset.loc[:, 'Sex'] = dataset['Sex'].map({'female': 0, 'male': 1})
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

svm = SVC()
svm.fit(df_train[features], df_train[target])
y_pred = svm.predict(df_test[features])

submission = pd.DataFrame({
    'PassengerId': df_test[passenger_id],
    'Survived': y_pred,
})
submission.to_csv('./submission.csv', index=False)