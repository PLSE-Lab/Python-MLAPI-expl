#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,roc_auc_score,explained_variance_score
from sklearn.metrics import accuracy_score,confusion_matrix, mean_absolute_error
import matplotlib.pyplot as plt


# **Reading the Data**

# In[ ]:


housing=pd.read_csv(r"../input/california-housing-prices/housing.csv")
housing.head()


# > Exploratory Data Analysis

# In[ ]:


housing.shape


# In[ ]:


housing.size


# In[ ]:


housing.describe()


# In[ ]:


housing.info()


# In[ ]:


housing.isna().sum()


# There are 207 data points not available in total bedrooms. Where the total number of bedrooms are the total number of bedrooms in all houses in the block.Lets drop the rows having null values.

# In[ ]:


housing=housing.dropna()
housing.isna().sum()


# In[ ]:


housing.shape


# In[ ]:


housing.size


# Now all the na values are dropped.

# In[ ]:


housing.median_house_value.describe()


# There is a column ocean_proximity(the type of landscape of the block) which is categorical variable. Lets convert it into numerical variable by label encoding. 

# There are 5 unique values in ocean_proximity.

# In[ ]:


print(housing.ocean_proximity.unique())
print(len(housing.ocean_proximity.unique()))


# In[ ]:


label_encoder=preprocessing.LabelEncoder()
housing.ocean_proximity= label_encoder.fit_transform(housing.ocean_proximity)
housing.ocean_proximity.unique()


# Here categorical column has changed to numerical column. The numbers were obtained based on alphabetical order.

# In[ ]:


housing.info()

From here, we can conform that all the columns were transformed into Numerical data.
# Standardizing the Dataset

# In[ ]:


names=housing.columns
scaler=preprocessing.StandardScaler()
scaled_housing=scaler.fit_transform(housing)
scaled_housing=pd.DataFrame(scaled_housing,columns=names)
scaled_housing.head()


# In[ ]:


X=scaled_housing.drop(columns=['median_house_value'])
X.head()


# In[ ]:


Y=scaled_housing['median_house_value']
Y=pd.DataFrame(Y)
Y.head()


# Predictor and response variables( independent and dependent variables) are divided. X represents Predictors and Y represents Response. We have to predict Y using X 

# > Splitting of Train and Test Data

# In[ ]:


train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.2,random_state=42)
print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)


# 80% training dataset and 20% test dataset 

# > Building Models

# Lets perform Multi Linear Regression 

# In[ ]:


linereg=LinearRegression()
model_linereg=linereg.fit(train_X,train_Y)
predict_y=model_linereg.predict(test_X)


# In[ ]:


print( 'Mean Square Error:',mean_squared_error(predict_y,test_Y))


# In[ ]:


print('Root Mean Square error:',np.sqrt(mean_squared_error(test_Y, predict_y)))


# In[ ]:


print('Mean absolute error:',mean_absolute_error(test_Y, predict_y))


# In[ ]:


# plot for residual error 
  
## setting plot style 
plt.style.use('fivethirtyeight') 
  
## plotting residual errors in training data 
plt.scatter(linereg.predict(train_X), linereg.predict(train_X) - train_Y, 
            color = "green", s = 10, label = 'Train data') 
  
## plotting residual errors in test data 
plt.scatter(linereg.predict(test_X), linereg.predict(test_X) - test_Y, 
            color = "blue", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
  
## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show() 


# In[ ]:


print ('Explaines Variance Score:',explained_variance_score(predict_y,test_Y))


# Now, Lets perform Decision Tree Regression 

# In[ ]:


dtreg=DecisionTreeRegressor(random_state=10)
dtreg_model=dtreg.fit(train_X,train_Y)
predict_y=dtreg.predict(test_X)


# In[ ]:


print('Mean Squared Error:',mean_squared_error(predict_y,test_Y))


# compared to linear regression mean square error , decision tree regression mean square error is some what less. But we can say the mean square errors are almost same

# In[ ]:


print('Root Mean Square error:',np.sqrt(mean_squared_error(test_Y, predict_y)))


# In[ ]:


print('Mean absolute error:',mean_absolute_error(test_Y, predict_y))


# In[ ]:


print ('Explaines Variance Score:',explained_variance_score(predict_y,test_Y))


# Ultimately, The error obtained using decision tree regressor are less than linear regression. But the change in error is negligible

# Extracting just the median_income column from the independent variables (from X_train and X_test)

# In[ ]:


X=scaled_housing['median_income']
X=pd.DataFrame(X)
X.head()


# In[ ]:


train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.2,random_state=42)
print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)


# Now lets perform Simple linear regression using only independent variable or predictor. Here the predictor we are using is median_income 

# In[ ]:


linereg=LinearRegression()
model_linereg=linereg.fit(train_X,train_Y)
predict_y=model_linereg.predict(test_X)


# In[ ]:


# plot for residual error 
  
## setting plot style 
plt.style.use('fivethirtyeight') 
  
## plotting residual errors in training data 
plt.scatter(linereg.predict(train_X), linereg.predict(train_X) - train_Y, 
            color = "green", s = 10, label = 'Train data') 
  
## plotting residual errors in test data 
plt.scatter(linereg.predict(test_X), linereg.predict(test_X) - test_Y, 
            color = "blue", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
  
## plotting legend 
plt.legend(loc = 'upper right') 
  
## plot title 
plt.title("Residual errors") 
  
## function to show plot 
plt.show() 


# In[ ]:


housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


# In[ ]:


print( 'Mean Square Error:',mean_squared_error(predict_y,test_Y))


# In[ ]:


print('Root Mean Square error:',np.sqrt(mean_squared_error(test_Y, predict_y)))


# In[ ]:


print('Mean absolute error:',mean_absolute_error(test_Y, predict_y))


# In[ ]:


print ('Explaines Variance Score:',explained_variance_score(predict_y,test_Y))


# In[ ]:


#housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, 
             label="population", c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
            )
plt.legend()


# This figure shows where the median_house_value is high and low
