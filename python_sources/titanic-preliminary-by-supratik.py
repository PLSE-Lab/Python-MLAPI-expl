# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVR
np.set_printoptions(formatter ={'float_kind':'{:.3f}'.format})
l=LabelEncoder()
r=LinearRegression()
sc=StandardScaler()
svr=SVR(kernel='rbf')
train = pd.read_csv("../input/train.csv").iloc[:,[1,4]]#1:Survived  4:Sex 
X= train.iloc[:,1:].values #0:Pclass 1:Sex 
Y=train.iloc[:,0:1].values
#print(X)                  #male=1,female=0


X[:,0] = l.fit_transform(X[:,0])
imp=Imputer(missing_values='NaN',strategy= 'mean',axis=0)
imp = imp.fit(X[:,0:1])
X[:,0:1] = imp.transform(X[:,0:1])
# Test Titanic Data Start

test = pd.read_csv("../input/test.csv")[0:100].iloc[:,[1,4]]
Xtest= test.iloc[:,1:].values
Ytest=test.iloc[:,0:1].values
Xtest[:,0] = l.fit_transform(Xtest[:,0])
# ENd
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.4,random_state=0)
#X_train= sc.fit_transform(X_train)
r.fit(X_train,Y_train)
svr.fit(X_train,Y_train)
#YN,"\n PRedict\n",YN_pred,
print(r2_score(Y_test,r.predict(X_test)))
print(r2_score(Ytest,svr.predict(Xtest)))
# after finding out which column has the shortest p value we storke it off
'''import statsmodels.formula.api as sm
X= np.append(arr = np.ones((100,1)).astype(int), values =X, axis=1)
X_ipy = X[:,[0,1,2,3,4]]#0 -2:pclass 3: Male/Female  4:Age
X_ipy = sm.OLS(endog=Y,exog=X_ipy).fit()
X_ipy.summary()
X_ipy = X[:,[1,3,4]]#0 -2:pclass 3: Male/Female  4:Age
X_ipy = sm.OLS(endog=Y,exog=X_ipy).fit()
X_ipy.summary()'''
plt.scatter(X_train,Y_train,color="Black")
plt.plot(X_train,r.predict(X_train),color="Blue")
plt.plot(X_train,svr.predict(X_train),color="Yellow")
plt.show()
# Any results you write to the current directory are saved as output.