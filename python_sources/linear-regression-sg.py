#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Contents
# 
# * [<font size=4>Getting Started</font>](#1)
#     * [Importing the Libraries](#1.1)
#     * [Importing and Inspecting the Data](#1.2)
#    
#    
# * [<font size=4>Fitting the model</font>](#2)
#     * [Setting up the input and the output variable](#2.1)
#     * [Fitting The linear Regression Model](#2.2)
#     * [Inspecting the Model](#2.3)
#     * [Checking out the validity of the model](#2.4)
#     
# 
# 

# # Getting Started <a id="1"></a>
# Here we describe importing the library, impoting the datset and some basic checks on the dataset

# # Import Libraries <a id="1.1"></a>

# In[ ]:


# lmport Libraries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm


# # Importing and Inspecting the Data <a id="1.2"></a>

# In[ ]:


dataset = pd.read_csv("/kaggle/input/Advertising.csv")
print(dataset.shape)
print(dataset.head(5))


# In[ ]:


dataset.describe()


# Scatter Plot

# In[ ]:


dataset.plot(x='Newspaper', y='Sales', style='o')  
plt.title('Sales and Newspaper Spend')  
plt.xlabel('TV')  
plt.ylabel('Sales')  
plt.show()


# # Fitting the Model <a id="2"></a>
# 
# In this section the input and output is setup, the train test split is done**

# # Setting up the input and the output variable <a id="2.1"></a>

# In[ ]:


# Selecting the Second, Third and Fouth Column
X= dataset.iloc[:,1:4]
# Selecting Fouth Columnn
y=dataset.iloc[:,4]


# # Fitting The linear Regression Model <a id="2.2"></a>

# In[ ]:


# Splitting the Data and output in training and testing
regressor = LinearRegression()  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor.fit(X_train, y_train)


# # Inspecting the Model <a id="2.3"></a>

# In[ ]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# In[ ]:


y_pred = regressor.predict(X_test)
df = pd.DataFrame({ 'Actual':y_test.values,'Predicted': y_pred})
ax1 = df.plot.scatter(x='Actual',
                      y='Predicted')


# # Checking out the validity of the model <a id="2.4"></a>

# In[ ]:


X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())

