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
data=pd.read_csv('../input/diamonds.csv')

#Drop col 1
del data['Unnamed: 0']
data['cut']=data['cut'].map({'Fair':0,'Good':1,'Very Good':2,'Premium':3,'Ideal':4})
data['color']=data['color'].map({'J':0,'I':1,'H':2,'G':3,'F':4,'E':5,'D':6})
data['clarity']=data['clarity'].map({'I1':0,'SI2':1,'SI1':2,'VS2':3,'VS1':4,'VVS2':5,'VVS1':6,'IF':7})
data['width_top']=(data['y']*data['table']) 
data = data.drop(['table'], axis=1) 
print(data.corr())

#split data
X=data.drop(['price'],axis=1) #all indepdendant variables
print(X.head())
Y=data['price'] #target variables
print(Y.head())

#splitting data as test and train, testing on 20% of the data and training on the 80.
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=np.random)

#linear regression
#for machine learning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn import linear_model

reg_all=linear_model.LinearRegression()

reg_all.fit(X_train,Y_train) #fitting the model for the x and y train

y_pred=reg_all.predict(X_test) #predicting y(the target variable), on x test

print(reg_all.score(X_test,Y_test))
print(reg_all.score(X_train,Y_train))
print(reg_all.score(X_test,y_pred))










"""
#List all the indepedent variables with the coeffecient from the linear regression model
coeff_df = pd.DataFrame(X_train.columns)
coeff_df.columns = ['Variable']
coeff_df["Coeff"] = pd.Series(reg_all.coef_)

coeff_df.sort_values(by='Coeff', ascending=True)
print(coeff_df)

#print intercept and rmse
print("Intercept: %f" %(reg_all.intercept_))
rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
print("rmse: %f" %(rmse))
"""