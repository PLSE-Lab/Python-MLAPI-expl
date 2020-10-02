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


# import library:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# import dataset:
df=pd.read_csv("../input/Position_Salaries.csv")


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


# dataset visualization:
sns.pairplot(df)


# In[ ]:


sns.countplot(df['Salary'],data=df)


# In[ ]:


sns.scatterplot(df['Level'],df['Salary'],data=df)


# In[ ]:


# spliting into the deopendent and independ dataset:
X=df.iloc[:,1:2].values
y=df.iloc[:,2].values


# In[ ]:


X


# In[ ]:


y


# In[ ]:


#spliting the datset into tarin and test:
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=(X,y,test_size=0.2,random_state=0)


# In[ ]:


# fitting the dataset on Poly_Regression:
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)


# In[ ]:


# fiting the datset on Polynomial Regression result
from sklearn.preprocessing import PolynomialFeatures
poly_regressor=PolynomialFeatures(degree=4)
X_poly=poly_regressor.fit_transform(X)
poly_regressor.fit(X_poly,y)
line_regressor=LinearRegression()
line_regressor.fit(X_poly,y)


# In[ ]:


# visulization on liner_regressor:
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Salar vs pokition')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()


# In[ ]:


# visualization on poly_regressor:
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,line_regressor.predict(poly_regressor.fit_transform(X_grid)),color='blue')
plt.title('Salary Vs position')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


# In[ ]:





# In[ ]:




