#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)


# In[ ]:


target = np.log(train.SalePrice)
plt.hist(target, color = 'blue')


# In[ ]:


train = train[train['GarageArea'] < 1200]


# In[ ]:


train['enc_street'] = pd.get_dummies(train.Street, drop_first = True)
test['enc_street'] = pd.get_dummies(test.Street, drop_first = True)


# In[ ]:


print(train.enc_street.value_counts())


# In[ ]:


print(test.enc_street.value_counts())


# In[ ]:


def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)


# In[ ]:


condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


data = train.select_dtypes(include=[np.number]).interpolate().dropna()


# In[ ]:


data.isnull().sum()


# In[ ]:


y = np.log(train.SalePrice)
X = data.drop(['SalePrice','Id'],axis = 1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42,test_size=.33)


# In[ ]:


from sklearn import linear_model
lr = linear_model.LinearRegression()


# In[ ]:


model = lr.fit(X_train,y_train)


# In[ ]:


model.score(X_train, y_train)


# In[ ]:


model.score(X_test, y_test)


# In[ ]:


predicts = model.predict(X_test) 


# In[ ]:


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,predicts))


# In[ ]:


actual_values = y_test
plt.scatter(predicts, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()


# In[ ]:


for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()


# In[ ]:


submission = pd.DataFrame()
submission['Id'] = test.Id


# In[ ]:


feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()


# In[ ]:


predictions = model.predict(feats)


# In[ ]:


final_predictions = np.exp(predictions)


# In[ ]:


print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])


# In[ ]:


submission['SalePrice'] = final_predictions
submission.head()


# In[ ]:


submission.to_csv('sub.csv', index=False)


# In[ ]:




