#!/usr/bin/env python
# coding: utf-8

# ## Import packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the data

# In[ ]:


data = pd.read_csv("../input/housing.csv")
data.head()


# In[ ]:


data.shape


# There are 20640 observations and 9 features + 1 target variable(median_house_value).

# In[ ]:


data.info()


# The above info function provide the information about the dataset .
# For example:
# * Missing values(no missing values in our dataset)
# * datatype(9 of them are floats and 1 is categorical)

# In[ ]:


# Pearson correlation
plt.subplots(figsize=(15, 9))
cor = data.corr()
sns.heatmap(cor, annot=True, linewidths=.5)
plt.show()


# If we have to select a single variable for the regression analysis then
# higher possibility is to pick the most correlated feature with the target variable(**median_house_value**).
# * In our case it is the **median_income** with correlation coefficent of **0.69**

# In[ ]:


# taking two variables
data = data.drop(["housing_median_age","households","total_bedrooms","longitude","latitude","total_rooms","population","ocean_proximity"], axis=1)
data.head()


# In[ ]:


X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]


# In[ ]:


plt.scatter(X, y, alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.show()


# Using this scatter plot we can infer that if a person has higher median_income then that person may have more expensive house.
# There is somewhat positive linear relationship between them.

# ## Split the data

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Model 1:

# ## Linear regression model

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Model initialization
regression_model = LinearRegression(normalize = True)

# Fit the data(train the model)
regression_model.fit(X_train, y_train)


# In[ ]:


# Predict
y_predicted = regression_model.predict(X_test)

# model evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
r2 = r2_score(y_test, y_predicted)


# In[ ]:


# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)


# >  **Interpretation**:
# 
# This simple linear regression with single variable (y = mx+b) has 
# * Slope of the line(m) : [42032.17769894]
# * Intercept (b) : 44320.63
# * R2 score:  0.4466 (For R2 score more is better in the range [0,1])
# * Root mean squared error:  84941.0515 (Lower is better)
# 

# **The plot of simple linear regression :**

# In[ ]:


# data points
plt.scatter(X_train, y_train, s=10)
plt.xlabel('median_income')
plt.ylabel('median_house_value')

# predicted values
plt.plot(X_test, y_predicted, color='r')
plt.show()


# ## Residual plot from linear regression

# In[ ]:


residual = y_test - y_predicted
sns.residplot(residual,y_predicted, lowess=True,scatter_kws={'alpha': 0.5},line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plt.show()


# **The residuals exhibit a clear non straight line, which provides a strong indication of non-linearity in the data**
# 
# This makes us to do somethhing more to find better fit of the model.
# > These are the possible approaches:
# 
#      First we will try some transformations to best fit the data.
#      Next could be try ploynomial regression of some higher degree to increase the R2 score.

# # Model 2:

# ## Applying transformation

# In[ ]:


tf = np.sqrt(X_train) 
tf1 = np.sqrt(X_test)

plt.scatter(tf, y_train)
plt.show()


# ## Fitting a model

# In[ ]:


regression_model.fit(tf, y_train)
# Predict
y_predicted = regression_model.predict(tf1)

# model evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
r2 = r2_score(y_test, y_predicted)

# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)


# >  **Interpretation**:
# 
# This transformed linear regression with single variable (y = mx+b) has 
# * Slope of the line(m) : 175550.81
# * Intercept (b) : -129097.46
# * R2 score:  0.4385 (For R2 score more is better in the range [0,1])
# 
# Found R2 score is worse than the simple linear regression. This suggest that we do not need  transformation in this case, we have try polynomial regression to improve the predication.
# * Root mean squared error:  85566.36 (Lower is better)
# 

# ## Residual plot for transformed

# In[ ]:


res = y_test - y_predicted
sns.residplot(res,y_predicted, lowess=True,scatter_kws={'alpha': 0.5},line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plt.show()


# Residual plot for the transformed linear regression is more zigzag than the simple linear regression.
# This residual plot suggest that transformation makes the relationship more non- linear in nature.

# # Model 3:

# ## Fitting polynomial Regression model

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression(normalize = True)
pol_reg.fit(X_poly, y_train)


# In[ ]:


def viz_polymonial():
    plt.scatter(X_train, y_train, color="red")
    plt.plot(X_train, pol_reg.predict(poly_reg.fit_transform(X_train)))
    plt.xlabel('median_income')
    plt.ylabel('median_house_value')
    plt.show()
    return
viz_polymonial()


# In[ ]:


# Predict
X_p = poly_reg.fit_transform(X_test)
y_predicted = pol_reg.predict(X_p)

# model evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
r2 = r2_score(y_test, y_predicted)

# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)


# >  **Interpretation**:
# 
# This transformed linear regression with single variable (y = mx+b) has 
# * Slope of the line(m) : 175550.81
# * Intercept (b) : -129097.46
# * R2 score:  0.4498 (For R2 score more is better in the range [0,1])
# 
# Found R2 score is the best so far. This means that we will keep this ploynomial model with degree 2 as our final and best model(but there is one other thing to consider i.e. simple is better than complex)
# * Root mean squared error:  84699.9 (Lower is better)

# In[ ]:


res = y_test - y_predicted
sns.residplot(res,y_predicted, lowess=True,scatter_kws={'alpha': 0.5},line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plt.show()


# ## Comparing the model

# * Model 1 has R2 score:  0.4466
# * model 2 has R2 score:  0.4385
# * model 3 has R2 score:  0.44982
# 
# **After analyzing the R2 score , My final model will be Model 1 as it is simple and has not worse R2 score as compared to the model 3.**
