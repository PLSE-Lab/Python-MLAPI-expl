# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from sklearn.svm import SVC
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import accuracy_score
from matplotlib.pylab import rcParams

current_file = os.path.abspath(os.path.dirname(__file__))
csv_filename = os.path.join(current_file, '../input/diabetes.csv')
train2 = pd.read_csv(csv_filename)

target = 'Outcome'

rcParams['figure.figsize'] = 12, 4

drop_elements = ['Outcome']

train2.loc[train2['Insulin'] <= 211.5, 'Insulin'] = 0
train2.loc[(train2['Insulin'] > 211.6) & (train2['Insulin'] <= 423), 'Insulin'] = 1
train2.loc[(train2['Insulin'] > 424) & (train2['Insulin'] <= 634.5), 'Insulin'] = 2
train2.loc[train2['Insulin'] > 634.5, 'Insulin'] = 3
#
train2.loc[train2['DiabetesPedigreeFunction'] <= 0.663, 'DiabetesPedigreeFunction'] = 0
train2.loc[(train2['DiabetesPedigreeFunction'] >= 0.664) & (train2['DiabetesPedigreeFunction'] <= 1.249), 'DiabetesPedigreeFunction'] = 1
train2.loc[(train2['DiabetesPedigreeFunction'] >= 1.250) & (train2['DiabetesPedigreeFunction'] <= 1.835), 'DiabetesPedigreeFunction'] = 2
train2.loc[(train2['DiabetesPedigreeFunction'] >= 1.836), 'DiabetesPedigreeFunction'] = 3

y = train2['Outcome'].values
train = train2.drop(drop_elements, axis=1)

X = train.iloc[:, :].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)

X_test_std = stdsc.fit_transform(X_test)

svm = SVC(kernel='linear', C=0.1, random_state=0)
svm.fit(X_train, y_train)
spred = svm.predict(X_test)
print ("Accuracy with SVM {0}".format(accuracy_score(spred, y_test) * 100))

