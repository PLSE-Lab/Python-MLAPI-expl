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


import sklearn.preprocessing as Prep
import sklearn.model_selection as mod
import sklearn.metrics as metrics


# Load Dataset 

# In[ ]:


training_data = pd.read_csv('../input/train.csv')
testing_data = pd.read_csv('../input/test.csv')


# In[ ]:


print("training data shape is : "+ str(training_data.shape))
print("testing data shape is : "+ str(testing_data.shape))


# In[ ]:


training_data.info()
print('='*50)
testing_data.info()


# In[ ]:


y = training_data['SalePrice'].values


# In[ ]:


training_data.head()


# In[ ]:


# data  = pd.concat([training_data, testing_data])
data = pd.concat([training_data.drop(['SalePrice'], axis=1), testing_data], axis=0)


# In[ ]:


data.shape
# data.head()


# In[ ]:


data.drop('Id', axis=1,inplace=True)
# testing_data.drop(['Id'], axis=1)
data.shape


# In[ ]:


data.head()


# In[ ]:


# # from sklearn.preprocessing import Imputer
# # imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# # imp.fit(training_data)
# # train = imp.transform(training_data)
for col in data:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].median())


# In[ ]:


# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# newdf = list(training_data.select_dtypes(include=numerics).columns.values)
# print(len(newdf))
data.shape


# In[ ]:


# from sklearn.preprocessing import  Imputer
# imputer = Imputer(strategy = 'median')

# imputer.fit(data)

# train_sample = imputer.transform(data)


# In[ ]:


print(data.isnull().sum().sum())


# In[ ]:


# data["Alley"]


# In[ ]:


X = pd.get_dummies(data)


# In[ ]:


X.head()


# In[ ]:


X.shape


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
result = scaler.fit_transform(X)


# In[ ]:


type(result)


# In[ ]:


brb2 = result[:training_data.shape[0]]
test_values = result[training_data.shape[0]:]


# In[ ]:


print(brb2.shape)
test_values.shape


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

features = X.iloc[:,0:289]  #independent columns
target = X.iloc[:,-1]    #target column i.e price range

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(features,target)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(features.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))


# In[ ]:


# import seaborn as sns
# import matplotlib.pyplot as plt
# corrmat = X.corr()
# top_corr_features = corrmat.index
# plt.figure(figsize=(80,80))
# g=sns.heatmap(X[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[ ]:


# train_label = X["SalePrice"]
# train_label.shape
# X


# In[ ]:


# from sklearn.model_selection import train_test_split

# X_train, X_test , y_train, y_test = train_test_split(X,train_label,train_size = 0.8)


# In[ ]:


# X_train.shape


# In[ ]:


# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# model.fit(X_train,y_train)


# In[ ]:


# y_preds = model.predict(X_test)


# In[ ]:


# from sklearn import metrics
# from sklearn.metrics import r2_score


# print("Root Mean square error: " , np.sqrt(metrics.mean_squared_error(y_test,y_preds)))
# print("Test acc: ", r2_score(y_test, y_preds))


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


X_train, X_test, y_train, y_test = train_test_split(brb2, y, random_state=42)

# clf = LinearRegression()
clf = Lasso()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)


# In[ ]:


from sklearn.metrics import r2_score

print("Train acc: " , r2_score(y_train, y_train_pred))
print("Test acc: ", r2_score(y_test, y_pred))


# In[ ]:


final_labels = clf.predict(test_values)


# In[ ]:


final_result = pd.DataFrame({'Id': testing_data['Id'], 'SalePrice': final_labels})


# In[ ]:


final_result.to_csv('house_price.csv', index=False)


# In[ ]:


print(os.listdir("../input"))


# In[ ]:





# In[ ]:




