#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #  **Import Libraries**

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Import Dataset

# In[ ]:


dataset = pd.read_csv('/kaggle/input/heights-and-weights/data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[ ]:


dataset.head()


# # Split Data into Training Set & Testing Set in 7:3 ratio

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


# # Train Simple Linear Regression Model

# In[ ]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# # Predict Data  

# In[ ]:


y_pred=regressor.predict(X_test)


# # Graph of Training data

# In[ ]:


plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Height vs Weight (Train Set)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show


# # Graph of Testing data

# In[ ]:


plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Height vs Weight (Test Set)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show


# # Checking Accuracy Score

# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

