# TODO
# Investigate whether a larger number of estimators has a positive effect?


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

train_data = pd.read_csv("../input/learn-together/train.csv", index_col='Id')
test_data = pd.read_csv("../input/learn-together/test.csv", index_col='Id')

X_train = train_data.drop('Cover_Type', axis='columns')
y_train = train_data['Cover_Type']

rf = RandomForestClassifier()
model = Pipeline(steps=[('model',rf),])

#Optimal parameters determined from the following:
#param_grid = {'model__n_estimators': np.logspace(0,3,5).astype(int),
#              'model__max_features': [0.1,0.3,0.5,0.7,0.9]}

#Setting them manually here for notebook speed and setting cv to 2 (was 3)
param_grid = {'model__n_estimators': [1000],
              'model__max_features': [0.3]}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, verbose=3)
grid.fit(X_train, y_train)
print('Best Score: {:.4f} ({:.4f}), Best Params: {}:'.format(grid.best_score_, 
                                                             grid.cv_results_['std_test_score'][grid.best_index_], 
                                                             grid.best_params_))
preds = grid.predict(test_data)
output = pd.DataFrame({'Id': test_data.index,
                       'Cover_Type': preds})
output.to_csv('submission.csv', index=False)