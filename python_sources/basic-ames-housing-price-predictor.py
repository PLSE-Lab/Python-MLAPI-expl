#!/usr/bin/env python
# coding: utf-8

# As of the date I am starting this kernel I am taking a neural networks course. This will be an attempt to apply some of the principles we have learned in the course up to this point. We are only part way through the semester and this is an intorductory course to the topic, so I do not expect the world's greatest model. The goal instead is to simply apply what has been learned so far and create a model which is not too difficult to explain. As more complexity is gradually added to the model, cross validation will be performed to see any changes in the results. To help keep the code relatively organized, the data will be loaded and any imports will be done in the first block. 

# In[ ]:


import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')


# To start off we will make a simple model using linear regression to relate the house prices to above ground square footage, but before we do so we will plot the data because this set is known for having outliers.

# In[ ]:


train.plot(x='GrLivArea', y='SalePrice', style='o')


# Clearly there are not many datapoints above 4000 square feet. The lectures have recently been delving into linear regression so I plan to use that for my first couple of attempts. Leaving these points would end up increasing the error when making predictions for the more common sized properties. So in order to minimize the overall error I will be removing these samples from the training dataset.

# In[ ]:


train = train[train.GrLivArea < 4000]
train.plot(x='GrLivArea', y='SalePrice', style='o')


# Visually the new plot looks like it could be a straight line or a parabola. We will try both fits using some portion of the data-set and compare the results with the unused portion. Although the course I have been taking has been delving into solving these problems using methods such as "Gradient Descent", the size of this dataset is not that large so I am opting to solve the equations using the pseudo-inverse of a matrix created by the square footage. I have shamelessly borrowed the RMSLE function from [here](http://www.kaggle.com/marknagelberg/rmsle-function).

# In[ ]:


#split the data using test-train split from sklearn.
X = train['GrLivArea']
d = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, d, test_size=0.33, random_state=1)

#Create the X matrix for y = W_0 + W_1*X where y is sale price and X is sq footage
X_1stOrder = np.row_stack((np.ones(X_train.size),X_train.values))
X_1stOrder_pinv = np.linalg.pinv(X_1stOrder)
W_1stOrder = np.dot(np.transpose(X_1stOrder_pinv),y_train)

#Add a row of ones to the test set to allow the matrix multiplication
X_test_1stOrder = np.row_stack((np.ones(X_test.size),X_test.values))
FirstOrderTestPredictions = np.dot(np.transpose(X_test_1stOrder),W_1stOrder)

#Create a function for RMSLE since this is what is used by the competition
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

FirstOrderError = rmsle(y_test.values,FirstOrderTestPredictions)
print(FirstOrderError)

#Next add a row of the GrLivArea^2 to the X_1stOrder to create X_2ndOrder
X_2ndOrder = np.row_stack((X_1stOrder, np.square(X_train.values)))
X_2ndOrder_pinv = np.linalg.pinv(X_2ndOrder)
W_2ndOrder = np.dot(np.transpose(X_2ndOrder_pinv),y_train)

#Similarly add a row to the test data-set
X_test_2ndOrder = np.row_stack((np.ones(X_test.size),X_test.values,np.square(X_test.values)))

SecondOrderTestPredictions = np.dot(np.transpose(X_test_2ndOrder),W_2ndOrder)
SecondOrderError = rmsle(y_test.values,SecondOrderTestPredictions)
print(SecondOrderError)


# Since the use of a second order does not seem to make much of a difference, going forward I am only going to use a first-order approximation for the above ground square footage to encourage a simpler model. Next, I will examine the basement square footage.

# In[ ]:


train.plot(x='TotalBsmtSF', y='SalePrice', style='o')


# This one looks more clearly like it is second order. Unfortunately, I am not sure how to deal with the houses which do not have a basement and hope someone will provide some guidance in the comments. My intuition says there are too many points to throw them away as outliers. Plus the model should be able to predict prices for properties which do not have basement prices as well. Until I have a better idea how to handle this case, I will simply include these in the training set.

# In[ ]:


X = train[['GrLivArea','TotalBsmtSF']]

#Again get a slice of data for training and a different one for cross-validation.
#Change the random state to make sure we are switching up the sets which are used
X_train, X_test, y_train, y_test = train_test_split(X, d, test_size=0.33, random_state=2)

#Build the appropriate X matrix
X_TwoVar = np.row_stack((np.ones(X_train['GrLivArea'].size),X_train['GrLivArea'].values,                  X_train['TotalBsmtSF'].values, np.square(X_train['TotalBsmtSF'].values)))
                        
X_pinv = np.linalg.pinv(X_TwoVar)
W_TwoVar = np.dot(np.transpose(X_pinv),y_train.values)

#Add a row of ones to the test set to allow the matrix multiplication
X_test_TwoVar = np.row_stack((np.ones(X_test['GrLivArea'].size),X_test['GrLivArea'].values,                  X_test['TotalBsmtSF'].values, np.square(X_test['TotalBsmtSF'].values)))
TwoVarTestPredictions = np.dot(np.transpose(X_test_TwoVar),W_TwoVar)
TwoVarError = rmsle(y_test.values,TwoVarTestPredictions)
print(TwoVarError)


# This brings the RMSLE down a fair amount. Based on other notebooks I have examined, it seems the OverallQual feature has a high correlation with the output price so we will examine this next.

# In[ ]:


train.plot(x='OverallQual',y='SalePrice', style='o')


# Visually speaking, this appears to have a "sort-of" parabolic shape, but the possible values of the independent variable are discrete. As of this moment my course has only begun getting into logistic regression, however, I do not believe this is applicable since we are not looking for a binary output. For the moment I am going to add this to the model in a manner similar to the square footage features I have previously examined. I am hoping that someone will provide an insightful comment that will help me to better understand how to integrate this sort of feature into the linear regression model I have put together at this point. Since it appears there would be a fair amount of variance, I am going to assume that this feature will simply end up having a smaller weight than it would otherwise.

# In[ ]:


X = train[['GrLivArea','TotalBsmtSF','OverallQual']]

#Again get a slice of data for training and a different one for cross-validation.
#Change the random state to make sure we are switching up the sets which are used
X_train, X_test, y_train, y_test = train_test_split(X, d, test_size=0.33, random_state=3)

#Build the appropriate X matrix
X_ThreeVar = np.row_stack((np.ones(X_train['GrLivArea'].size),X_train['GrLivArea'].values,                  X_train['TotalBsmtSF'].values, np.square(X_train['TotalBsmtSF'].values),                  X_train['OverallQual'].values, np.square(X_train['OverallQual'].values)))
                        
X_pinv = np.linalg.pinv(X_ThreeVar)
W_ThreeVar = np.dot(np.transpose(X_pinv),y_train.values)

#Add a row of ones to the test set to allow the matrix multiplication
X_test_ThreeVar = np.row_stack((np.ones(X_test['GrLivArea'].size),X_test['GrLivArea'].values,                  X_test['TotalBsmtSF'].values, np.square(X_test['TotalBsmtSF'].values),                  X_test['OverallQual'].values, np.square(X_test['OverallQual'].values)))
ThreeVarTestPredictions = np.dot(np.transpose(X_test_ThreeVar),W_ThreeVar)
ThreeVarError = rmsle(y_test.values,ThreeVarTestPredictions)
print(ThreeVarError)


# Based upon my own experience looking at houses the lot area tends to be important. Looking at the plot, it seems there may be some value to applying a linear fit if we ignore some of the really large lots (there are not a lot of those anyhow). This will likely result in some error for larger lots, probably farms, but will probably decrease the error on the smaller lot sizes.

# In[ ]:


train.plot(x='LotArea',y='SalePrice', style='o')

train = train[train.LotArea < 60000]
d = train['SalePrice']

train.plot(x='LotArea',y='SalePrice', style='o')

#Try a logarithmic fit
X = train[['GrLivArea','TotalBsmtSF','OverallQual','LotArea']]

#Again get a slice of data for training and a different one for cross-validation.
#Change the random state to make sure we are switching up the sets which are used
X_train, X_test, y_train, y_test = train_test_split(X, d, test_size=0.33, random_state=4)

#Build the appropriate X matrix
X_FourVar = np.row_stack((np.ones(X_train['GrLivArea'].size),X_train['GrLivArea'].values,                  X_train['TotalBsmtSF'].values, np.square(X_train['TotalBsmtSF'].values),                  X_train['OverallQual'].values, np.square(X_train['OverallQual'].values),                  X_train['LotArea'].values))
                        
X_pinv = np.linalg.pinv(X_FourVar)
W_FourVar = np.dot(np.transpose(X_pinv),y_train.values)

#Add a row of ones to the test set to allow the matrix multiplication
X_test_FourVar = np.row_stack((np.ones(X_test['GrLivArea'].size),X_test['GrLivArea'].values,                  X_test['TotalBsmtSF'].values, np.square(X_test['TotalBsmtSF'].values),                  X_test['OverallQual'].values, np.square(X_test['OverallQual'].values),                  X_test['LotArea'].values))
FourVarTestPredictions = np.dot(np.transpose(X_test_FourVar),W_FourVar)
FourVarError = rmsle(y_test.values,FourVarTestPredictions)
print(FourVarError)


# In[ ]:


#Get the desired features from the test set
X = test[['GrLivArea','TotalBsmtSF','OverallQual','LotArea']]
X = np.row_stack((np.ones(X['GrLivArea'].size),X['GrLivArea'].values,                  X['TotalBsmtSF'].values, np.square(X['TotalBsmtSF'].values),                  X['OverallQual'].values, np.square(X['OverallQual'].values),                  X['LotArea'].values))

#There seem to be missing data with respect to TotalBsmtSF. So we can actually
#solve the problem and get submittable results, we will replace all nan with zeros
X = np.nan_to_num(X)
Predictions = np.dot(np.transpose(X),W_FourVar)

#Get the Id for the properties
Ids = test['Id']
nanRow = np.argwhere(np.isnan(Predictions))


submissionDF = pd.DataFrame({'Id':Ids,'SalePrice':Predictions})
submissionDF.head()
submissionDF.to_csv('results.csv',index=False)

print(train.columns)

