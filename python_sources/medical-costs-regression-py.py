#!/usr/bin/env python
# coding: utf-8

# # Predicting Medical Costs Using Regression

# ## Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn import metrics
style.use('fivethirtyeight')


# # Dataset

# ## Importing the Dataset

# In[ ]:


medical_DF = pd.read_csv("../input/insurance/insurance.csv")


# In[ ]:


medical_DF.head(10)


# In[ ]:


medical_DF.shape


# There are 1338 rows and 7 columns in this dataset.

# # Checking for Missing Values

# In[ ]:


medical_DF.isnull().sum()


# There are no missing values in each column.

# # Exploratory Data Analysis

# In[ ]:


f, ax1 = plt.subplots(2, 2, figsize = (10,10))
sns.countplot(data = medical_DF, x = "sex", ax = ax1[0,0])
sns.countplot(data = medical_DF, x = "children", ax = ax1[0,1])
sns.countplot(data = medical_DF, x = "smoker", ax = ax1[1,0])
sns.countplot(data = medical_DF, x = "region", ax = ax1[1,1])
plt.show()


# * **Sex**: The amount of males and females are almost equal. 50.5% of all beneficiaries are male.
# * **Children**: 42.9% of all insurance holders have no children. 
# * **Smoker**: 79.5% of all beneficiaries are non-smokers.
# * **Region**: The distribution of regions are almost equal, with the South-East region having the most beneficiaries.

# In[ ]:


f, ax2 = plt.subplots(2, 2, figsize = (12,10))
sns.violinplot(data = medical_DF, x = "age", ax = ax2[0,0], color = '#d43131')
sns.violinplot(data = medical_DF, x = "bmi", ax = ax2[0,1], color = '#3152d2')
sns.violinplot(data = medical_DF, x = "charges", ax = ax2[1,0], color = '#bd7a17')
f.delaxes(ax = ax2[1,1])
plt.show()


# In[ ]:


medical_DF[['age', 'bmi', 'charges']].describe()


# * Age: The average age is 39.2 years. The median age is 39 years. The lowest recorded is 18 and the highest is 64. There is a huge distribution of beneficiaries between the ages 20 and 30.
# * BMI: The average BMI is 30.66. The median BMI is 30.4. The lowest recorded BMI is 15.96 while the highest recorded BMI is 53.13.
# * Charges: The average medical cost is 13,270.42 US dollars. The median medical cost is 9382.03 dollars. The minimum medical cost is 1121.87 dollars. The maximum is 63,770.43 dollars.

# ## Medical Charges in terms of Sex

# In[ ]:


plt.figure(figsize = (10,5))
sns.violinplot(data = medical_DF, x = "charges", y = "sex", hue = "sex")
plt.show()


# In[ ]:


medical_DF.groupby(['sex']).describe()['charges']


# The median medical charges for females are slightly higher than males. However, the average medical costs for males are slightly higher. 

# ## Medical Charges in terms of Smokers and Non-Smokers

# In[ ]:


plt.figure(figsize = (10,5))
sns.violinplot(data = medical_DF, x = "charges", y = "smoker", hue = "smoker")
plt.show()


# In[ ]:


medical_DF.groupby(['smoker']).describe()['charges']


# The medical charges for smokers are significantly higher. The average medical cost for smokers is 32,050.23 dollars whereas the average cost for non-smokers is 8434.27 dollars. The minimum medical cost for a non-smoker is 1121.87 dollars while it is 12,829.46 for smokers. The highest cost incurred by a non-smoker is 36,910.61 dollars while the highest bill incurred by a smoker is 63,770.43 dollars.

# ## Medical Charges in terms of Region

# In[ ]:


plt.figure(figsize = (10,10))
sns.violinplot(data = medical_DF, x = "charges", y = "region", hue = "region")
plt.show()


# In[ ]:


medical_DF.groupby(['region']).describe()['charges']


# The South-East region has the highest average medical costs while the South-West region has the lowest average medical costs. The North-East region has the highest median medical cost while the South-West region has the lowest median medical cost.

# ## Medical Charges in terms of Number of Children

# In[ ]:


plt.figure(figsize = (10,10))
sns.violinplot(data = medical_DF, x = "charges", y = "children", hue = "children")
plt.show()


# In[ ]:


medical_DF.groupby(['children']).describe()['charges']


# Beneficiaries with 4 children have the highest median medical charges while beneficiaries with 1 child have the lowest median medical charges. Beneficiaries with 5 children have the lowest average medical costs (which can be attributed to their small population) while beneficiaries with 3 children have the highest average medical bills.

# ## Medical Charges vs BMI

# In[ ]:


plt.figure(figsize = (6,6))
sns.scatterplot(data = medical_DF, x = "charges", y = "bmi")
plt.show()


# There is a weak correlation between the medical charges and the BMI of the beneficiaries.

# ## Medical Charges vs Age

# In[ ]:


plt.figure(figsize = (6,6))
sns.scatterplot(data = medical_DF, x = "charges", y = "age")
plt.show()


# There is a weak correlation between the medical charges and the age of the beneficiaries.

# # Correlation

# In[ ]:


med_corr = medical_DF.corr()
plt.figure(figsize = (7,7))
sns.heatmap(med_corr, annot = True, linewidths = 1.2, linecolor = 'white')
plt.show()


# None of the continuous features have strong correlations with each other.

# # Dummy Variables

# There are many categorical features in this dataset. To perform regression, I will convert them to dummy variables.

# In[ ]:


medical_DF_1 = medical_DF


# In[ ]:


medical_DF_1 = pd.get_dummies(medical_DF, columns = ['region', 'children', 'sex'])
medical_DF_1.drop(columns = ['sex_female'], inplace = True)
medical_DF_1["smoker"].replace({"yes": 1, "no": 0}, inplace=True)


# In[ ]:


medical_DF_1.head(10)


# In[ ]:


med_corr = medical_DF_1.corr()
plt.figure(figsize = (14,10))
sns.heatmap(med_corr, annot = True, linewidths = 1.2, linecolor = 'white')
plt.show()


# As seen in the correlogram above, being a smoker has a high correlation to medical charges. Most of the features have weak correlations with each other.

# # Normalization

# Normalization scales the data in such a way that the values range from 0 to 1.

# In[ ]:


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

medical_DF_1["age"] = NormalizeData(medical_DF_1["age"])
medical_DF_1["bmi"] = NormalizeData(medical_DF_1["bmi"])

medical_DF_1.head(10)


# # Linear Regression

# In[ ]:


X = medical_DF_1.drop(columns=['charges'])
Y = medical_DF_1['charges']


# ## Train and Test Split

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# ## Model

# In[ ]:


med_reg = LinearRegression()
med_reg.fit(X_train, Y_train)


# ## Using the Model to Predict

# In[ ]:


med_pred = med_reg.predict(X_test)


# ## Actual vs Predicted Values

# In[ ]:


med_pred_DF = pd.DataFrame({'Actual': Y_test, 'Predicted': med_pred})
med_pred_DF.head(20)


# ## R-Squared Value

# R-squared value measures how well the values are predicted. The higher the R-squared value, the better the model fits the data.

# In[ ]:


metrics.r2_score(Y_test, med_pred)


# This model explains 78.26% of the variation in the response variable.

# ## MAE, MSE and RMSE Values

# Mean Absolute Error measures the absolute value of the errors between the actual values and the predicted values.
# 
# <img src = "https://miro.medium.com/max/1040/1*tu6FSDz_FhQbR3UHQIaZNg.png" width = 300px>
# 
# Mean Squared Error measures the average squared difference between the predicted values and the actual values.
# 
# <img src = "https://i.imgur.com/vB3UAiH.jpg" width = 300px>
# 
# Root Mean Squared Error measures the square root of the differences between predicted and actual values.
# 
# <img src = "https://secureservercdn.net/160.153.137.16/70j.58d.myftpupload.com/wp-content/uploads/2019/03/rmse-2.png" width = 300px>

# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, med_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, med_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, med_pred)))


# # Polynomial Regression

# Polynomial regression is a special case of linear regression where we fit a polynomial equation on the data with a curvilinear relationship between the target variable and the independent variables. 
# 
# Equation for Linear Regression and Polynomial Linear Regression ([source of image](http://medium.com/@subarna.lamsal1/multiple-linear-regression-sklearn-and-statsmodels-798750747755)):
# 
# <img src = "https://miro.medium.com/max/2086/1*XSBSL7LbDOvjXyi4wz-i_g.png" width = 600px>

# In[ ]:


X_poly = PolynomialFeatures(degree = 2, include_bias = False).fit_transform(X)


# PolynomialFeatures() generates a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree (in this case, the degree is 2).

# ## Train and Test Split

# In[ ]:


X_poly_train, X_poly_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.2, random_state=42)


# ## Model

# In[ ]:


med_poly_reg = LinearRegression().fit(X_poly_train, Y_train)


# ## Using the Model to Predict

# In[ ]:


med_poly_pred = med_poly_reg.predict(X_poly_test)


# ## R-Squared Value

# In[ ]:


metrics.r2_score(Y_test, med_poly_pred)


# The polynomial model explains 86.44% of the variation in the response variable.

# ## MAE, MSE and RMSE Values

# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, med_poly_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, med_poly_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, med_poly_pred)))


# # Conclusion

# The Polynomial Regression model yields better results, with lower error rates and better R-squared values.
