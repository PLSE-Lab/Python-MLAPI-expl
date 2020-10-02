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
import os 
dir_name=os.path.abspath(os.path.dirname(__file__))
data_file='../input/kc_house_data.csv'
data_file_path=os.path.join(dir_name, data_file)
import pandas
table=pandas.read_csv(data_file_path)
import numpy
features=numpy.array(table)[:,3:]
Y=numpy.array(table)[:,2]

# (2) Split
# Training: Testing = 80:20
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
features_train, features_test, Y_train, Y_test =(train_test_split(features, Y, test_size=0.20))
                        
'[1] Ridge Regression with CV over alpha'
Reg = linear_model.RidgeCV(alphas=[0.001,0.01, 1.0, 10.0, 100])
Reg=Reg.fit(features_train,Y_train)
predicted_Y=Reg.predict(features_test)
print ('Ridge Regression')
print ('Best alpha is '), Reg.alpha_  
print('mae: %.2f' % mean_absolute_error(Y_test,predicted_Y))
print("Mean squared error: %.2f" % numpy.mean((predicted_Y - Y_test) ** 2))
print('Variance score: %.2f' % Reg.score(features_test, Y_test))

'[2] Lasso with CV over alpha'
Reg = linear_model.LassoCV(alphas=[0.001,0.01, 1.0, 10.0, 100])
Reg=Reg.fit(features_train,Y_train)
predicted_Y=Reg.predict(features_test)
print ('Lasso')
print ('Best alpha is '), Reg.alpha_ 
print('mae: %.2f' % mean_absolute_error(Y_test,predicted_Y)) 
print("Mean squared error: %.2f" % numpy.mean((predicted_Y - Y_test) ** 2))
print('Variance score: %.2f' % Reg.score(features_test, Y_test))


'[3] ElasticNet with CV over alpha and l1_ratio'
Reg = linear_model.ElasticNetCV(alphas=[0.001,0.01, 1.0, 10.0, 100],
            l1_ratio=[0.001,0.01, 1.0, 10.0, 100])
Reg=Reg.fit(features_train,Y_train)
predicted_Y=Reg.predict(features_test)
print ('Elastic Net')
print ('Best alpha is '), Reg.alpha_  
print ('Best l1_ratio is '), Reg.l1_ratio_ 
print('mae: %.2f' % mean_absolute_error(Y_test,predicted_Y)) 
print("Mean squared error: %.2f" % numpy.mean((predicted_Y - Y_test) ** 2))
print('Variance score: %.2f' % Reg.score(features_test, Y_test))