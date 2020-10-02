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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
test[test>0]=1

X_train = train.drop('label', axis = 1)
X_train[X_train>0]=1
y_train = train['label']

#X = train.drop('label', axis = 1)
#y = train['label']
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
#print(model.score(X_test, y_test))

predictions = model.predict(test)
np.savetxt('first_svm.csv', np.c_[range(1, len(test) + 1), predictions], delimiter=',', comments = '', header = 'ImageId,Label', fmt='%d')
'''
from sklearn.grid_search import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.0001,0.000001,0.000001,0.0000000001]}
grid = GridSearchCV(SVC(),param_grid,verbose = 3)
grid.fit(X_train.head(2000),y_train.head(2000))
grid.best_params_
grid_predictions = grid.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, predictions))
print('\n')
print(confusion_matrix(y_test, predictions))
'''