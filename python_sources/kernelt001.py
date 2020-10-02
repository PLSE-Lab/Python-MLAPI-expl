#====================================================================================================
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# author: Ahmed Hamada
#====================================================================================================    


import numpy as np
import pandas as pd



dataset_train=pd.read_csv('../input/machathon-10-filteration-test/train.csv')
dataset_test=pd.read_csv('../input/machathon-10-filteration-test/test.csv')

predict = 'TotalOilInNext6Months'
X = np.array(dataset_train.drop([predict], 1)) # Features
y = np.array(dataset_train[predict]) # Label
X.shape
y.shape


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# X_train = dataset_train.iloc[:,:].values
# Y_train = dataset_test.iloc[:,-1].values
# Y_test.shape
# Y_pred.shape
# X_test = dataset_test.iloc[:,1:-1].values


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 

print('Random Forest Regression    [30, 300]')
regressor = RandomForestRegressor(max_depth=30, random_state=0, n_estimators = 300)
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_train)
    
MAEValue = mean_absolute_error(Y_train, Y_pred, multioutput='uniform_average') 
print('Mean Absolute Error Value is : ', MAEValue)

MSEValue = mean_squared_error(Y_train, Y_pred, multioutput='uniform_average')
print('Mean Squared Error Value is : ', MSEValue)

print('==========================================================')







