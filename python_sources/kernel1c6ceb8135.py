# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as AS
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv("../input/glass.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, 9]
X_train, X_test, Y_train, Y_test = tts(X, y)
# Any results you write to the current directory are saved as output.

model = LR()
model.fit(X_train, Y_train)
print(AS(model.predict(X_test), Y_test) * 100)
