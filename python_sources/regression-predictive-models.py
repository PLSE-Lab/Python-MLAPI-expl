#!/usr/bin/env python
# coding: utf-8

# Haiii Kagglers, in this kernel i will show you how to deal with regression problem.
# Today i will be using 3 algorithm, Linear regression, Lasso CV, and Ridge CV.
# let's get started.

# ## Import Modules

# In[ ]:


# Import Modules

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
from scipy.special import inv_boxcox
from scipy.stats import boxcox


# In[ ]:


# read File

Data_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
Data_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
Submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')


# ## Quick Look
# Lets take a quick look at the dataset

# In[ ]:


Data_train.head()


# In[ ]:


Data_test.head()


# In[ ]:


Submission.head()


# In[ ]:


Data_train.isnull().sum()


# In[ ]:


Data_train.info()


# In[ ]:


Data_test.info()


# There is 1460 rows in total, and we also got few columns with missing value. So i'm going to drop few column with missing value more than 50%, but first let's make sure they are not important feature to our target column.
# Let's check the correlation.

# In[ ]:


#Correlation map to see how features are correlated with SalePrice
Cor_heat = Data_train.corr()
plt.figure(figsize=(16,16))
sns.heatmap(Cor_heat, vmax=0.9, square=True)


# ## Data Cleansing

# In[ ]:


# Drop Missing value
# I drop these because they had so many missing value and lower corr with sale price

Columns = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

Data_train = Data_train.drop(Columns, axis=1)
Data_train.head()


# ## Feature Engineering
# 
# After performing cleansing, we have to encode the categorical data and fill some missing value on the dataset

# In[ ]:


# Encode Categorical features to numerical features with get dummies from pandas

Data_train = pd.get_dummies(Data_train)
Data_train = Data_train.fillna(method='ffill')
Data_train.head()


# Lets take a look at the distribution in our target column, Sale Price.

# In[ ]:


## Basic Distribution

print('Skew Value : ' + str(Data_train.SalePrice.skew()))
sns.distplot(Data_train.SalePrice)


# Skew value is 1.8, which is bad, this could affect the accuracy in the models. lets Transform this column and lets see if we can find the closest skew value to zero (the best distribution)

# In[ ]:


f = plt.figure(figsize=(16,16))

# log 1 Transform
ax = f.add_subplot(221)
L1p = np.log1p(Data_train.SalePrice)
sns.distplot(L1p,color='b',ax=ax)
ax.set_title('skew value Log 1 transform: ' + str(np.log1p(Data_train.SalePrice).skew()))

# Square Log Transform
ax = f.add_subplot(222)
SRT = np.sqrt(Data_train.SalePrice)
sns.distplot(SRT,color='c',ax=ax)
ax.set_title('Skew Value Square Transform: ' + str(np.sqrt(Data_train.SalePrice).skew()))

# Log Transform
ax = f.add_subplot(223)
LT = np.log(Data_train.SalePrice)
sns.distplot(LT, color='r',ax=ax)
ax.set_title('Skew value Log Transform: ' + str(np.log(Data_train.SalePrice).skew()))

# Box Cox Transform
ax = f.add_subplot(224)
BCT,fitted_lambda = boxcox(Data_train.SalePrice,lmbda=None)
sns.distplot(BCT,color='g',ax=ax)
ax.set_title('Skew Value Box Cox Transform: ' + str(pd.Series(BCT).skew()))


# The best distribution value is with Box Cox Transform with value -0.008, this is the best distribution for our target dataset. But just to make sure we are going to perform validation score with all these transformation.

# In[ ]:


## Lets see what most important features we have

IF = Cor_heat['SalePrice'].sort_values(ascending=False).head(10).to_frame()
IF.head(4)


# ## Building Models
# 
# Let's try building a models

# In[ ]:


# make Models

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# In[ ]:


# Split The data

Train = Data_train.drop('SalePrice', axis=1)
Test = Data_train.SalePrice


# In[ ]:


# Assign the distribution of Sale Price

feature_SP = {'Log Transform': LT,
              'Square Root Transform': SRT,
              'Box-Cox Transform':BCT,
              'Log 1 Transform': L1p}


# In[ ]:


from sklearn.model_selection import cross_val_score, cross_val_predict

# Perform 5-fold CV

reg = LinearRegression()
lcv = LassoCV()
rcv = RidgeCV()

alg = [reg, lcv, rcv]
    
for y in alg:
    print(str(y.__class__.__name__) + ' results')
    for key, value in feature_SP.items():
        Test = value
        score = np.sqrt(-cross_val_score(y, Train, Test, scoring='neg_mean_squared_error', cv=5))
        print('RMSE with ' + str(key) + ' : ' + str(np.mean(score)))


# The lowest RMSE score is 0.05 with RidgeCV algorithm and Box Cox Transform, this is the combination we are going use to perform prediction in test dataset.

# ## Perform Prediction
# 
# Let's treat the test data in the same way we treat train data by droping few columns, Performing Feature Engineering and fill some missing value.

# In[ ]:


Columns = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

Data_test = Data_test.drop(Columns, axis=1)
Data_test.head()

# Fill Missing Value
Data_test = pd.get_dummies(Data_test)
Data_test = Data_test.fillna(method='ffill')
Data_test.head()


# In[ ]:


Train.head()


# There is something wrong here, the total of columns are different in both dataset, we cant perform any prediction like this. Let's check what columns is in train dataset but not in test dataset. 

# In[ ]:


d1 = Train.columns
d2 = Data_test.columns

out = []

for x in d1:
    if x in d2:
        pass
    else:
        out.append(x)

print(out)
print(len(out))


# I'm not sure what are these columns, but i think the dataset generate this when we perform encoding to our dataset, don't worry and just drop them.

# In[ ]:


Data_trains = Train.drop(out, axis=1)
Data_trains.shape


# Now the train dataset have 254 columns, it the same count as test dataset. Now let's perform prediction on the test dataset, remember the best algorithm is ridge CV and the best transformation is Box Cox Transformation, let's apply those rule here.

# In[ ]:


# Use data trains as train
Train = Data_trains
# Best Alg = Ridge CV
model = rcv
# X_train
X_train = Train
# Y_train = Box Cox Sale Price
Test = BCT
y_train = Test
# X_test
X_test = Data_test
# Y_pred
model.fit(X_train, y_train)
y_pred = inv_boxcox(model.predict(X_test), fitted_lambda)
y_pred


# In[ ]:


Submission['SalePrice'] = y_pred
Submission.head()


# In[ ]:


Submission.to_csv("Final Submission File.csv",index=False)


# ## End
# 
# That is how you handle regression problem.
# Thank you :)
