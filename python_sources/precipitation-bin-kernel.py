# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df = pd.read_excel(io=r"../input/weatherOrg.xlsx")

print(df.describe())

to_train = df.drop(['Date', 'Location', 'RainTomorrow'], axis=1)
outcome = df[['RainTomorrow']]

categorical_elect = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

numeric_to_scale = to_train.drop(categorical_elect, axis=1)
numeric_to_scale.fillna(0, inplace=True)

ss = StandardScaler()

ss.fit(numeric_to_scale)
numeric_fields = numeric_to_scale.columns
numeric_to_scale[numeric_to_scale.columns] = ss.transform(numeric_to_scale)

numeric_to_scale[categorical_elect] = to_train[categorical_elect]
to_train_v2 = numeric_to_scale

to_train_v2 = pd.get_dummies(to_train_v2, columns=categorical_elect)
print(to_train_v2)

new_fields = list(set(to_train_v2.columns) - set(numeric_fields))

lr = LogisticRegression()
model = lr.fit(to_train_v2, np.array(outcome))

outcome_labels = model.predict(to_train_v2)
now_labels = np.array(outcome['RainTomorrow'])


g = sns.FacetGrid(to_train_v2, col='RainToday_Yes')
g.map(sns.pointplot, 'RainToday_No')
g.add_legend()

print('Accuracy:', float(accuracy_score(now_labels, outcome_labels))*100, '%')
print('Classification Stats:')
print(classification_report(now_labels, outcome_labels))