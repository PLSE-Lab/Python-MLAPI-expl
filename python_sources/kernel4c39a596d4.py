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
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../input/BlackFriday.csv')

agelist=list(dataset.iloc[:,3].values)
avgage=[i.split("-") for i in agelist]
avgage=[(int(i[1])+int(i[0]))/2 if(i[0][-1] != '+') else int(i[0][0:2]) for i in avgage]



X = dataset.iloc[:, 3:11].values
X[:,0]=avgage
y = dataset.iloc[:, 11].values


#missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean' , axis = 0)
imputer = imputer.fit(X[:, [-1,-2] ])
X[: ,[-1,-2] ] = imputer.transform(X[:, [-1,-2] ])

#to remove +4
X[:,3] = [int(i[:-1]) if(i[-1] == '+') else int(i[:]) for i in X[:,3] ]


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,2] =labelencoder_X.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[2])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]



#statsmodel
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((537577,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0, 1, 2, 3, 4, 5,6 ,7]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()#"""check max p value and remove"""
X_opt=X[:,[0, 1, 2, 3, 4, 6,7]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0, 3]]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.20, random_state = 0)



#fitting linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)




