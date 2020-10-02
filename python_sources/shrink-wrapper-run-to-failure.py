# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

worn_data = []
new_data = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if 'New' in filename:
            new_data.append(pd.read_csv(os.path.join(dirname, filename)))
        else:
            worn_data.append(pd.read_csv(os.path.join(dirname, filename)))
        print(os.path.join(dirname, filename))

x_worn_data = pd.concat(worn_data)
x_new_data = pd.concat(new_data)

y_worn_data = ['worn' for i in range(len(x_worn_data))]
y_new_data = ['new' for i in range(len(x_new_data))]

X = pd.concat([x_worn_data, x_new_data])
y = y_worn_data + y_new_data
print("Imported datasets")
print()

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
clf = LinearSVC(random_state=42,
                max_iter=10000,
                dual=False 
               )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
#pred = clf.predict(X_test)

print(clf.score(X_test, y_test))