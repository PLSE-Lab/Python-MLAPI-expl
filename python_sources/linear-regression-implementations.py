# Goal: implementing linear regression models from scratch
# (it is a good programming, and interesting to see that there are many ways to do the very same thing)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#let's consider just LotArea as X for the moment

X=np.array(train['LotArea'])
y=np.array(train['SalePrice'])

# 1st implementation: normal equations (using numpy)
# 2nd implemetation: QR decomposition
# 3rd implementation: SVD decomposition.
# they are all ways to solve for beta the equation
# X*beta=Y
# where X is (n,p), beta is (p,1), Y is (n,1)
class CustomLinearRegression():
    def __init__(self):
        self.coef_=None
        
    def fit(self, trainX, trainY, method):
        trainX_ = np.stack((trainX,np.ones(np.shape(trainX)[0])),axis=0).transpose()
        if method == 'normal':
            self.coef_=np.dot(np.dot(np.linalg.inv(np.dot(trainX_.transpose(),trainX_)),trainX_.transpose()),trainY)
        if method == 'QR':
            q, r = np.linalg.qr(trainX_)
            self.coef_=np.dot(np.dot(np.linalg.inv(r),q.transpose()),trainY)
        if method == 'SVD':
            U, s, V = np.linalg.svd(trainX_, full_matrices=False)
            #np.dot(
            self.coef_=np.dot(np.dot(V.transpose(),np.dot(np.linalg.inv(np.diag(s)),U.transpose())),trainY)
            #,trainY)
        
    def predict(self, testX):
        return np.dot(self.coef_,np.stack((testX,np.ones(np.shape(testX)[0])),axis=0))

lm=CustomLinearRegression()
lm.fit(X,y,'normal')
print("Custom coefficients - OLS standard method: " + str(lm.coef_))
lm2=CustomLinearRegression()
lm2.fit(X,y,'QR')
print("Custom coefficients - QR decomposition method: " + str(lm2.coef_))
lm3=CustomLinearRegression()
lm3.fit(X,y,'SVD')
print("Custom coefficients - SVD decomposition method: " + str(lm3.coef_))

#now let's check that coefficients are the same of the true Linear Regression (i.e., sci-kit learn implementation, 
# which in the case of OLS is just a wrap-up of a scipy function)
from sklearn.linear_model import LinearRegression
skl=LinearRegression()
skl.fit(np.reshape(X,(1460,1)),y)
print("sklearn coefficients: " + str([skl.coef_,skl.intercept_]))

#they are, thankfully.
skl.predict(np.reshape(X,(1460,1)))
lm.predict(X)

#The kernel is a working progress. I'll add new estimation procedure/improve code quality soon. 
# Good readings:
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/base.py
# https://stats.stackexchange.com/questions/1829/what-algorithm-is-used-in-linear-regression
