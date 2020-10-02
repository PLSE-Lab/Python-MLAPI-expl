#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# * In this notebook:
#     * Custom dataset creation for playing with different scenarios
#     * Comparison of the result of Linear Regression, Feature engeeniring (X^2), Polynomial Regression, and RandomForest regression 

# #### Database
# 
# * the data are created using a True function. So the solution is already in the function and there would no need to use any ML techniques. This is a study using the dummy data to check the impact of polynomial features on the performance of a linear regression and also to study the features importance in RandomForest  Regressor.

# In[ ]:


np.random.seed(0)

n_samples = 400

def true_fun(X_1, X_2 , X_3):
    
    y_temp_1 = np.cos(1.5 * np.pi * X_1)
    y_temp_2 = y_temp_1 * X_2 * X_2
    y_temp_3 = X_3 * 0.5
    
    result = 2 * y_temp_1 + y_temp_2 + y_temp_3
    return result

X_1 = np.sort(np.random.rand(n_samples))
X_2 = np.sort(np.random.rand(n_samples)) +  np.random.randn(n_samples) * 0.2

data = pd.DataFrame(X_1, columns= {"X_1"})
data["X_2"] = X_2
data["X_3"] = 1

data["X_3"].loc[data["X_2"] >0.4] = 0.5
y = true_fun(data["X_1"], data["X_2"], data["X_3"]) + np.random.randn(n_samples) * 0.2

data["y"] = y


# ### Plot the data

# In[ ]:


sns.pairplot(data, kind="reg" , diag_kind="kde")


# ### Split the Data

# In[ ]:


X = data.drop("y" , axis = 1)
y = data["y"]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# ## Training the Model
# 
# #### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
print (lm.score(X_train,y_train))
print (lm.score(X_test,y_test))


# ### A bit of feature engeeniring

# In[ ]:


data["X_4"] = data["X_1"] * data["X_1"]


# In[ ]:


X = data.drop("y" , axis = 1)
y = data["y"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
print (lm.score(X_train,y_train))
print (lm.score(X_test,y_test))


# ### PolynimialFeatures solution

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
polynomial_features = PolynomialFeatures(degree=3,include_bias=False)
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
pipeline.fit(X_train,y_train)
predictions = pipeline.predict( X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

print (pipeline.score(X_train,y_train))
print (pipeline.score(X_test,y_test))


# ### Random Foresest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100, max_features = 4)
rf.fit(X_train,y_train)
print(regr.feature_importances_)
predictions = rf.predict( X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
print (rf.score(X_train,y_train))
print (rf.score(X_test,y_test))


# ### Conclusion
# 
# * PolynomialFeatures method captures the 3 grade degree of the function that maps X to y
# * RandomForests Regressor captures in particular the 4th feature which makes sense if I look how that feature influences the output

# In[ ]:




