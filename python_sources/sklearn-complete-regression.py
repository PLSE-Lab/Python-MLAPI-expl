# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from  sklearn import model_selection
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

dataset=pd.read_csv('../input/ML.csv', names=['Population','Profit'])
array=dataset.values
print (array[0:10])

#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
dataset.plot(kind='scatter' , x='Population',  y='Profit')
plt.show()


X=array[:,:1]
Y=array[:,1]

seed=7
X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y,test_size=0.20,random_state=seed)

print (X[0:10])
print (Y[0:10])
# Spot Check Algorithms
models = []
models.append(('SGDRegressor', linear_model.SGDRegressor(alpha=0.01,max_iter=1000)))
models.append(('linear Regression', linear_model.LinearRegression()))
models.append(('Ridge Regression', linear_model.Ridge(alpha = .5)))
models.append(('Lasso Regression',linear_model.Lasso(alpha = 0.1)))


scoring='neg_mean_squared_error'
for name,model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print (msg)
# Any results you write to the current directory are saved as output.