import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
dataset=pd.read_csv("../input/winequality-red.csv")


x=dataset.iloc[:,0:11].values
y=dataset.iloc[:,11].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
y_train=y_train.reshape(-1,1)
y_train=sc_y.fit_transform(y_train)
y_test=y_test.reshape(-1,1)
y_test=sc_y.transform(y_test)


#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x, y)

y_pred1 = regressor.predict(x_test)
y_pred1=y_pred1.astype(np.int64)

#XGBoost
from xgboost import XGBRegressor
regressor=XGBRegressor()
regressor.fit(x_train,y_train)

#Kfold cross validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=regressor , X=x_train , y = y_train , cv=10)
accuracies.mean()
accuracies.std()



import os
print(os.listdir("../input"))
