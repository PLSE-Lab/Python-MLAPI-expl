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

#read the input file 
dataset = pd.read_csv('../input/cereal.csv')

#See the top 5 rows to visualize the data
dataset.head()

#describe the dataset
dataset.describe()

#dataset info
dataset.info()

#remove name column 
dataset.drop(['name'],axis=1,inplace=True)
dataset = pd.get_dummies(dataset)

#remove name and rating columns from the dataset and assign all other remaing columns to X
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values


X = dataset[['calories','protein','fat','sodium','fiber','carbo','sugars','potass','vitamins','shelf','weight','cups','rating','mfr_A','mfr_G','mfr_K','mfr_N','mfr_P','mfr_Q','type_C']]
Y = dataset['rating']



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=0,test_size=0.2)


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
#y_test = y_test.reshape(-1,1)

y_predicted = regressor.predict(X_test)

#regressor.score(X_test, y_test)
#from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test, y_predicted))

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 5,scoring='neg_mean_squared_error')
print('')
print('####### LinearRegression #######')
print('Score : %.4f' % regressor.score(X_test, y_test))
print(f'The Mean of the model is {accuracies.mean()} and the standard deviation is {accuracies.std()}')

mse = mean_squared_error(y_test, y_predicted)
mae = mean_absolute_error(y_test, y_predicted)
rmse = mean_squared_error(y_test, y_predicted)**0.5
r2 = r2_score(y_test, y_predicted)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)






#Apply Label Encode for manufacturer column
#from sklearn.preprocessing import LabelEncoder
#labelencoder = LabelEncoder()
#X[:, 0] = labelencoder.fit_transform(X[:, 0])


#Apply One Hot Encoder for manufacturer column
#from sklearn.preprocessing import OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()

#one_hot_encoded_training_predictors = pd.get_dummies(X[:,0])



