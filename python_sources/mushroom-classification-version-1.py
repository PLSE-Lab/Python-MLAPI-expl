# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import f1_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
mushrooms = pd.read_csv("../input/mushrooms.csv")

data = mushrooms[["class", "odor", "habitat","population", "spore-print-color", "gill-color", "ring-type", "stalk-color-below-ring"]]
label_encoder = LabelEncoder()
categorical_columns = data.columns[data.dtypes == 'object']
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])
    
amount = int(len(data) * 0.8)
train_data = data[0:amount][[ "odor", "habitat","population", "spore-print-color", "gill-color", "ring-type", "stalk-color-below-ring"]].values
train_labels = data[0:amount][[ "class"]].values

test_data = data[amount:][[ "odor", "habitat","population", "spore-print-color", "gill-color", "ring-type", "stalk-color-below-ring"]].values
test_labels = data[amount:][[ "class"]].values


clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_labels)

result = clf.predict(test_data)
f1_res = f1_score(test_labels, result)
print("F1 score = " + str(f1_res))
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.