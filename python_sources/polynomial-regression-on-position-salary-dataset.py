#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df=pd.read_csv('/kaggle/input/position-salaries/Position_Salaries.csv')


print(df)


# In[ ]:


df.info()


# In[ ]:


df['Position'].value_counts()


# In[ ]:


plt.scatter(df['Level'],df['Salary'])


# In[ ]:


X=df.iloc[:,1].values.reshape(-1,1)
y=df.iloc[:,-1].values

X


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


X_train


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression

L=LinearRegression()


# In[ ]:


L.fit(X_train , y_train)


# In[ ]:


y_pred = L.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error

print('The r2 score',r2_score(y_test,y_pred))
print('The rmse',np.sqrt(mean_squared_error(y_test,y_pred)))
print('The mean absolute error',mean_absolute_error(y_test,y_pred))


# In[ ]:


plt.plot(X_test,y_pred,color='red')
plt.scatter(X,y)
plt.legend()
plt.show()


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 5)


# In[ ]:


X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.fit_transform(X_test)


# In[ ]:


print(X_train_poly)





# In[ ]:


print(X_test_poly)


# In[ ]:


L.fit(X_train_poly,y_train)


# In[ ]:


y_pred_2 = L.predict(X_test_poly)


# In[ ]:


print('The r2 score',r2_score(y_test,y_pred_2))
print('The rmse',np.sqrt(mean_squared_error(y_test,y_pred_2)))
print('The mean absolute error',mean_absolute_error(y_test,y_pred_2))


# In[ ]:


poly_2 = PolynomialFeatures(degree = 10)
X_train_poly_2=poly_2.fit_transform(X_train)
X_test_poly_2=poly_2.fit_transform(X_test)
L.fit(X_train_poly_2,y_train)
y_pred_3 = L.predict(X_test_poly_2)



print('The r2 score',r2_score(y_test,y_pred_3))
print('The rmse',np.sqrt(mean_squared_error(y_test,y_pred_3)))
print('The mean absolute error',mean_absolute_error(y_test,y_pred_3))


# In[ ]:


poly_3 = PolynomialFeatures(degree = 8)
X_train_poly_3=poly_3.fit_transform(X_train)
X_test_poly_3=poly_3.fit_transform(X_test)
L.fit(X_train_poly_3,y_train)
y_pred_4 = L.predict(X_test_poly_3)



print('The r2 score',r2_score(y_test,y_pred_4))
print('The rmse',np.sqrt(mean_squared_error(y_test,y_pred_4)))
print('The mean absolute error',mean_absolute_error(y_test,y_pred_4))


# In[ ]:


poly_4 = PolynomialFeatures(degree = 7)
X_train_poly_4=poly_4.fit_transform(X_train)
X_test_poly_4=poly_4.fit_transform(X_test)
L.fit(X_train_poly_4,y_train)
y_pred_5 = L.predict(X_test_poly_4)



print('The r2 score',r2_score(y_test,y_pred_5))
print('The rmse',np.sqrt(mean_squared_error(y_test,y_pred_5)))
print('The mean absolute error',mean_absolute_error(y_test,y_pred_5))


# In[ ]:


poly_5 = PolynomialFeatures(degree = 6)
X_train_poly_5=poly_5.fit_transform(X_train)
X_test_poly_5=poly_5.fit_transform(X_test)
L.fit(X_train_poly_5,y_train)
y_pred_6 = L.predict(X_test_poly_5)



print('The r2 score',r2_score(y_test,y_pred_6))
print('The rmse',np.sqrt(mean_squared_error(y_test,y_pred_6)))
print('The mean absolute error',mean_absolute_error(y_test,y_pred_6))


# In[ ]:



plt.scatter(X_test,y_pred_6,color='red')
plt.scatter(X,y,marker='+',color='green')
plt.legend()
plt.show()


# #   **Hence the polynomial degree should be 6 for the above dataset as it is giving the maximum r2score and minimum MAE and RMSE   **

# In[ ]:





# 

# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




