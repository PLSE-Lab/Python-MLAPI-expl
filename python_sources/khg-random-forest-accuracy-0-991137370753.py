# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

##import modules##
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics 
import matplotlib.pyplot as plt

##load_data##
data = pd.read_csv('../input/mushrooms.csv')

##split target, dependent_variables
target = data['class']
target = pd.DataFrame(target)
train = data.iloc[:, 1:,]

# string converts to One_Hot_Encoding
le = LabelEncoder()
train = train.apply(le.fit_transform)
target = target.apply(le.fit_transform)

## train, test
def split(train, target):
    x_train, x_test, y_train, y_test = train_test_split(train, target)
    
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = split(train, target)

##random_forest##
random_forest = RandomForestClassifier(n_estimators=100, max_depth = 5, random_state=1)
model = random_forest.fit(x_train, y_train)
importances = model.feature_importances_

#graph
indices = np.argsort(importances)
features = train.columns
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()

#prediction, result
print("test :", model.score(x_train, y_train))
prediction = model.predict(x_test)
print("accuracy :", metrics.accuracy_score(prediction, y_test))