#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


path =  '/kaggle/input/usa-housingcsv'
os.chdir(path)


# In[ ]:


data = pd.read_csv('USA_Housing.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# I select price as the dependent variable and the rest are independent variables. <br>This means I predict the price based on the independent varibles

# Now it is time to do some visualisations with the data.

# In[ ]:


sns.pairplot(data)


# ## Now, lets find the correlation between these variables.
# The correlation coefficient, or simply the correlation, is an index that ranges from **-1 to 1**. When the value is **near zero, there is no linear relationship**. As the correlation gets **closer to plus or minus one, the relationship is stronger**. A value of **one (or negative one) indicates a perfect linear relationship between two variables.**

# In[ ]:


sns.heatmap(data.corr())


# From the heatmap, it seems like **Price and Avg. Area Income** has a **strong linear relationship.**

# # Splitting the data
# Let's split the data into training and testing data. X will be the features to train on, and Y will be the target variable.
# <br>
# x = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
#                'Avg. Area Number of Bedrooms', 'Area Population']
# <br>y = Price                

# In[ ]:


X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']


# In[ ]:


from sklearn.model_selection import train_test_split


# ## Test Train Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# **test_size = 0.4**, means that 40% of the data goes to the test data and the r[](http://)est remains in the training set.

# # Train the Model

# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[ ]:


LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)


# In[ ]:


lm.coef_


# # Predict the model

# In[ ]:


predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# Our model is pretty okay

# # Evaluating the Model
# Let's calculate some error

# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


r2_score(y_test, predictions)


# In[ ]:


# exclude avg area no of bedroom
X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                'Area Population']]
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Instantiate model
lm2 = LinearRegression()

# Fit Model
lm2.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

# Predict
y_pred = lm2.predict(X_test)

# RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Variance Score
print(r2_score(y_test, y_pred))


# # Conclusion

# In[ ]:


coeffecients = pd.DataFrame(lm2.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# These numbers mean that holding all other features fixed, a 1 unit increase in Avg. Area Income will lead to an increase in $21.528348 in Price, and similarly for the other features
