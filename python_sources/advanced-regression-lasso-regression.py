#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from math import log
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# First let's define functions we are going to use. There are three functions:
# 
# 1. miss_values: to fill in miss values
# 
# 2. factor_encoding: to transfer factor variables into dummy variables
# 
# 3. log_skew: log transform columns that are skewed

# In[ ]:


def miss_values(df):
    for column in df:
        # Test whether column has null value
        if len(df[column].apply(pd.isnull).value_counts()) > 1:
            print(column+" has missing value")
            #if column is numeric, fill null with mean
            if df[column].dtype in ('int64','float64'):
                df[column] = df[column].fillna(df[column].mean())
            else:
                df[column] = df[column].fillna("unknown")


# In[ ]:


def log_skew(df):
    for column in df:
        if df[column].dtype in ('int64','float64') and column != 'SalePrice':
            old_skew = df[column].skew()
            if abs(df[column].skew()) > 1.0:
                df[column] = df[column].apply(lambda x: log(x+1,2))
                print('the skewness of '+column+" is reduced from "+                      str(old_skew) + " to "+str(df[column].skew()))


# In[ ]:


def factor_encoding(df):
    for column in df:
        if df[column].dtype == 'object':
            df = df.merge(pd.get_dummies(data=df[column],prefix=column),right_index=True,left_index=True)
            del df[column]
    return df


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train['source'] = 1.0
test['source'] = 0.0
alls = pd.concat([train,test],ignore_index=True)
miss_values(alls)
log_skew(alls)
alls = factor_encoding(alls)


# In[ ]:


alls[0:5]


# In[ ]:


train = alls[alls['source'] == 1.0]
train.drop(['source'],inplace=True,axis=1)
test = alls[alls['source'] == 0.0]
test.drop(['source'],inplace=True,axis=1)


# In[ ]:


x = train.drop(['SalePrice'],axis=1)
Y = train['SalePrice']
x_test = test.drop(['SalePrice'],axis=1)
x_train, x_vali, Y_train, Y_vali = train_test_split(x, Y, test_size=0.25, random_state=42)


# In[ ]:


# Lasso Regression
alpha = [i*0.1 for i in range(0,30)]
mse_lasso_test = [0]*len(alpha)
mse_lasso_train = [0]*len(alpha)
para_retained_lasso = [0]*len(alpha)
for i,al in enumerate(alpha):
    clf_lasso = linear_model.Lasso(alpha=al,max_iter=10000,normalize=True)
    clf_lasso.fit(x_train,Y_train)
    Y_predict_lasso = clf_lasso.predict(x_vali)
    para_retained_lasso[i] = len([i for i in list(clf_lasso.coef_) if i > 0.0])
    mse_lasso_train[i] = mean_squared_error(Y_train,clf_lasso.predict(x_train))**0.5
    mse_lasso_test[i] = mean_squared_error(Y_vali,Y_predict_lasso)**0.5


# In[ ]:


print(mse_lasso_test[0:5])
print(mse_lasso_train[0:5])
print(para_retained_lasso[0:5])


# In[ ]:


line_lasso_test, = plt.plot(alpha,mse_lasso_test,'g',label='Lasso(test)')
line_lasso_train, = plt.plot(alpha,mse_lasso_train,'g--',label='Lasso(train)')
plt.legend(handles=[line_lasso_test,line_lasso_train])
plt.xlabel('Hiper-parameter: Alpha')
plt.ylabel('Square-Root MSE')
plt.title('Tuning the hiper paremeter Alpha(Lasso)')
plt.show()


# In[ ]:


para_retain, = plt.plot(alpha,para_retained_lasso,'r--',label='retained parameters')
plt.legend(handles=[para_retain])
plt.xlabel('Alpha')
plt.ylabel('Number of parameters')
plt.title('Number of parameters retained')
plt.show()


# In[ ]:


clf = linear_model.Lasso(alpha=20,max_iter=10000,normalize=True)
clf.fit(x,Y)
test_predicted = clf.predict(x_test)


# In[ ]:


test['SalePrice'] = test_predicted
result = test[['Id','SalePrice']]
result.to_csv('submission.csv',index=False)

