#!/usr/bin/env python
# coding: utf-8

# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#8a3ebe;">Intro ...
#     
# This note book will cover the basic of Linear Regression(shrinkage) model and performance comparison of its type 
# * Ridge Regression
# * Lasso Regression
# 
# These will help us to avoid the overfitting problem by avoid "[curse of dimensionality](https://www.kdnuggets.com/2017/04/must-know-curse-dimensionality.html)" in linear model.
# 

# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#8a3ebe;">**Dataset**
# 
# The dataset has been taken from the UCI Machine Learning Repository. It can be found here: https://archive.ics.uci.edu/ml/datasets/Auto+MPG
# 
# The aim is to predict the mpg attribute. The dataset contains the following variables:
# 
# * mpg: continuous
# * cylinders: multi-valued discrete
# * displacement: continuous
# * horsepower: continuous
# * weight: continuous
# * acceleration: continuous
# * model year: multi-valued discrete
# * origin: multi-valued discrete
# * car name: string (unique for each instance)

# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#8a3ebe;">Load required Libs

# In[ ]:


# Numerical libraries
import numpy as np

# to handle data in form of rows and columns 
import pandas as pd

from sklearn import preprocessing

# Import Linear Regression machine learning library
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score


#importing ploting libraries
import matplotlib.pyplot as plt

#importing seaborn for statistical plots
import seaborn as sns


# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#8a3ebe;">Data Preprocessing 

# In[ ]:


#read the data and store it in data frame
df = pd.read_csv('/kaggle/input/autompg-dataset/auto-mpg.csv')
df


# In[ ]:


#the car name feature is not helping so we can drop it.
df = df.drop('car name',axis=1)


# In[ ]:


#just drop the Nan records
df.dropna()


# In[ ]:


#replace special character or junk data
df = df.replace('?',np.nan)


# In[ ]:


#replace the Nan value with the Mean value
df = df.apply(lambda x: x.fillna(x.median()),axis=0)


# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#8a3ebe;">Model Building Process
# 

# In[ ]:


#Separate independent and dependent variables

# Copy all the predictor variables into X dataframe. Since 'mpg' is dependent variable drop it
X = df.drop('mpg',axis=1)

# Copy the 'mpg' column alone into the y dataframe. This is the dependent variable
y = df[['mpg']]


# #Separate independent and dependent variablesScaling the features

# In[ ]:


# scale all the columns of df. This will produce a numpy array
X_scaled = preprocessing.scale(X)
X_scaled = pd.DataFrame(X_scaled,columns=X.columns)

y_scaled = preprocessing.scale(y)
y_scaled = pd.DataFrame(y_scaled,columns=y.columns)


# Now split the data into two for train model and test model.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled,test_size=0.30,random_state=1)


# here fit a simple  linear regression model

# In[ ]:


reg_model = LinearRegression()
reg_model.fit(X_train,y_train)


# Now print and check the coefficient of our Regression model

# In[ ]:


for idx,col_name in enumerate(X_train.columns):
    print('Coefficient for {} is {}'.format(col_name,reg_model.coef_[0][idx]))


# In[ ]:


intercept = reg_model.intercept_[0]
print('The intercept of the model is {}'.format(intercept))


# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#8a3ebe;">Create a RIDGE model and print the coefficients

# * The option "alpha" is called regularization term .Please read here for more info [Regularization ](https://www.geeksforgeeks.org/regularization-in-machine-learning/)
# * The regularization term "alpha" helps to prevent/stop the co-efficient to become high.
# * The value of alpha should not be high or low

# In[ ]:


ridge = Ridge(alpha=0.3)
ridge.fit(X_train,y_train)

print('Ridge model',(ridge.coef_))


# * Observe that the co-efficients are now changed little bit from the coefficient of origional model.

# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#8a3ebe;"> Create a LASSO model and print the coefficients

# In[ ]:


lasso = Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
print('Lasso coefficeient :',lasso.coef_)


# * **Observe:**
# * many of the coefficients have become 0 indicating drop of those dimensions from the model
# * Lasso minimise the co-efficeint to Zero but ridge reduce it fractionaly 
# * Lasso removed/droped 5 dimesion which its thinks those are useless.

# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#8a3ebe;">  score Comparison

# In[ ]:


print("Linear Regression Training score is {}".format(reg_model.score(X_train,y_train)))
print("Linear Regression Training score is {}".format(reg_model.score(X_test,y_test)))


# In[ ]:


print("Redge Training model score is {}".format(ridge.score(X_train,y_train)))
print("Redge Test model score is {}".format(ridge.score(X_test,y_test)))


# In[ ]:


print("Lasso Training model score is {}".format(lasso.score(X_train,y_train)))
print("Lasso Test model score is {}".format(lasso.score(X_test,y_test)))


# * More or less similar results but with less complex models.  Complexity is a function of variables and coefficients
# * Note - with Lasso, we get equally good result in test though not so in training. 
# * Further, the **number of dimensions is much less in LASSO**  model than ridge or un-regularized model
# * Lasso give good results with only 5 dimensions but Ridge and regular regression give the same with 10 dimensions
# * Over all Lasso model will survie in production well since its use less dimensions because with more dimensions we may endup with Overfit issue

# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#8a3ebe;"> Play With Polynomial models 
# 
# * Polynomial function will take existing dimesions and understand the relationship bewtween those dimensions and will generate new dimensions.([power of polynomial](https://acadgild.com/blog/polynomial-regression-understand-power-of-polynomials))
# 

# In[ ]:


#import the required library
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2,interaction_only=True)


# * degree: the option "degree" used to mention how many new feature need to create and the value is "2" which means that function will create new dimension where its origional dimension raised to power 2 only.
# * interaction_only: the "interaction_only" option set to True whic means the function will consider the dimensions with relationships between there.
# 

# In[ ]:


X_poly = poly.fit_transform(X_scaled)
X_poly.shape


# * using poly funtion we arrived with 29 dimensions 

# In[ ]:


#create train and test model using new X dataframe    

X_poly = poly.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.30, random_state=1)
X_test.shape


# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#8a3ebe;"> Fit non regularized linear model on poly features

# In[ ]:


reg_model.fit(X_train,y_train)
print(reg_model.coef_[0])


# * The model sholud have peaks and valley which is known by magnitudite of coefficient which tends to overfit problem

# <span style="font-family: Arial; font-weight:bold;font-size:1.5em;color:#8a3ebe;"> Play with Ridge model with Poly feature

# In[ ]:


ridge = Ridge(alpha=0.3)
ridge.fit(X_train,y_train)
print("Ridge model coefficient is {}".format(ridge.coef_))


# In[ ]:


print("Train model accuracy :" ,
      ridge.score(X_train,y_train))
print("Test model accuracy :" ,
      ridge.score(X_test,y_test))


# In[ ]:


lasso = Lasso(alpha=0.3)
lasso.fit(X_train,y_train)
print("Lasso model coefficient is {}".format(lasso.coef_))


# * We can clearly see that the Lasso model makes some of coefficient as zero because model thinks that those are not useful dimensions

# In[ ]:


print("Train model accuracy :" ,
      lasso.score(X_train,y_train))
print("Test model accuracy :" ,
      lasso.score(X_test,y_test))


# * By observing results above  its clear that "Lasso" giving good performance like Redge but with less number of dimensions
# * So Lasso model is ised in many situation for **Dimensionality Reduction or Feature Selection**.

# In[ ]:




