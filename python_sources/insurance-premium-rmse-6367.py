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


insurance_df = pd.read_csv('../input/insurance.csv')


# In[ ]:


insurance_df.head(5)


# In[ ]:


insurance_df.shape


# In[ ]:


insurance_df.info()


# In[ ]:


insurance_df.describe()


# In[ ]:


insurance_df.isna().sum()


# In[ ]:


list(insurance_df)


# In[ ]:


insurance_df[insurance_df.duplicated()]


# In[ ]:


insurance_df=insurance_df.drop_duplicates()


# In[ ]:


import seaborn as sns


# In[ ]:


ax = sns.boxplot(x=insurance_df["bmi"])


# In[ ]:


ax = sns.boxplot(x=insurance_df["expenses"])


# In[ ]:


ax = sns.boxplot(data=insurance_df)


# # Above plots shows only expenses and bmi have outliers

# In[ ]:


lower_bnd = lambda x: x.quantile(0.25) - 1.5 * ( x.quantile(0.75) - x.quantile(0.25) )


# In[ ]:


upper_bnd = lambda x: x.quantile(0.75) + 1.5 * ( x.quantile(0.75) - x.quantile(0.25) )


# # Removing expenses outliers

# In[ ]:


insurance_df = insurance_df[(insurance_df['expenses'] >= lower_bnd(insurance_df['expenses'])) & (insurance_df['expenses'] <= upper_bnd(insurance_df['expenses'])) ] 


# In[ ]:


insurance_df.shape


# # Removing bmi outliers

# In[ ]:


insurance_df = insurance_df[(insurance_df['bmi'] >= lower_bnd(insurance_df['bmi'])) & (insurance_df['bmi'] <= upper_bnd(insurance_df['bmi'])) ] 


# In[ ]:


insurance_df.shape


# In[ ]:


insurance_df.corr()


# In[ ]:


sns.pairplot(insurance_df)


# In[ ]:


sns.scatterplot(x=insurance_df['bmi'],y=insurance_df['expenses'],hue=insurance_df['smoker'])


# In[ ]:


insurance_df['region'].unique()


# In[ ]:


sns.pairplot(insurance_df,hue = 'smoker')


# In[ ]:


sns.pairplot(data=insurance_df,hue = 'sex')


# In[ ]:


sns.pairplot(data=insurance_df,hue = 'region')


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
insurance_df.iloc[:, 1] = labelencoder.fit_transform(insurance_df.iloc[:, 1])
insurance_df.iloc[:, 4] = labelencoder.fit_transform(insurance_df.iloc[:, 4])
insurance_df.iloc[:, 5] = labelencoder.fit_transform(insurance_df.iloc[:, 5])


# In[ ]:


insurance_df.corr()


# In[ ]:


from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score


# In[ ]:


insurance_df.head(5)


# In[ ]:


modelinput = insurance_df.drop(columns=['expenses','region'],axis=1)
modeloutput = insurance_df['expenses']


# In[ ]:


from sklearn import preprocessing
modelinput = preprocessing.StandardScaler().fit(modelinput).transform(modelinput.astype(float))


# In[ ]:


X_train,X_test,Y_train, Y_test = train_test_split(modelinput,modeloutput,test_size=0.3,random_state=123)


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,Y_train)


# In[ ]:


print("Intercept value:", lm.intercept_)
print("Coefficient values:", lm.coef_)


# In[ ]:


Y_train_predict = lm.predict(X_train)
Y_test_predict = lm.predict(X_test)


# In[ ]:


print("MSE Train:",mean_squared_error(Y_train, Y_train_predict))
print("MSE Test:",mean_squared_error(Y_test, Y_test_predict))


# In[ ]:


print("RMSE Train:",np.sqrt(mean_squared_error(Y_train, Y_train_predict)))
print("RMSE Test:",np.sqrt(mean_squared_error(Y_test, Y_test_predict)))


# In[ ]:


print('MAE Train', mean_absolute_error(Y_train, Y_train_predict))
print('MAE Test', mean_absolute_error(Y_test, Y_test_predict))


# In[ ]:


print('R2 Train',r2_score(Y_train, Y_train_predict))
print('R2 Test',r2_score(Y_test, Y_test_predict))


# In[ ]:


from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_dict = {}
accuracy_list = []
for K in range(50):
    K_value = K + 1
    neigh = KNeighborsRegressor(n_neighbors=K_value,weights='uniform',algorithm='auto')
    neigh.fit(X_train, Y_train)
    y_pred=neigh.predict(X_test)
    accuracy = r2_score(Y_test, y_pred)
    accuracy_dict.update({K_value:accuracy})
    accuracy_list.append(accuracy)
    print("Accuracy is",r2_score(Y_test, y_pred)*100,"% for K-Value",K_value)


# In[ ]:


key_max = max(accuracy_dict.keys(), key=(lambda k: accuracy_dict[k]))

print( "The Accuracy value is ",accuracy_dict[key_max], "with k= ", key_max)


# In[ ]:


elbow_curve = pd.DataFrame(accuracy_list,columns = ['accuracy'])
elbow_curve.plot()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
dt = DecisionTreeRegressor(max_depth=4) 
dt.fit(X_train, Y_train)
y_pred = dt.predict(X_test)
print(r2_score(Y_test, y_pred)*100)


# Depth

# In[ ]:


for i in range(1, 20):
    print('Accuracy score using max_depth =', i, end = ': ')
    dt = DecisionTreeRegressor(max_depth=i)
    dt.fit(X_train, Y_train)
    y_pred = dt.predict(X_test)
    print(r2_score(Y_test, y_pred)*100)


# Features

# In[ ]:


for i in ['auto','sqrt','log2',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    print('Accuracy score using max_features =', i, end = ': ')
    dt = DecisionTreeRegressor(max_depth=4,max_features=i)
    dt.fit(X_train, Y_train)
    y_pred = dt.predict(X_test)
    print(r2_score(Y_test, y_pred)*100)


# min_samples_split

# In[ ]:


for i in range(2, 10):
    print('Accuracy score using min_samples_split =', i, end = ': ')
    dt = DecisionTreeRegressor(max_depth=4,max_features=0.8,min_samples_split=i)
    dt.fit(X_train, Y_train)
    y_pred = dt.predict(X_test)
    print(r2_score(Y_test, y_pred)*100)


# In[ ]:




