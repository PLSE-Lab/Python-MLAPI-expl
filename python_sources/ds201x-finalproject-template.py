#!/usr/bin/env python
# coding: utf-8

# # General Steps to be followed for an end-to-end Machine Learning Project

# # 1. Define the Problem/Outcome

# - What is the question and what are we trying to solve/find out?
# - What are the generation intentions behind doing this project and how is it going to be helpful?

# # 2. Prepare the dataset

# - One of the major issues with working with datasets is that they are not always formatted and in a way, we want. We need to clean up the data in various ways. We need to make sure that the data we have is properly formatted and without any issues. We might need to add some data and remove some data, which can be done by viewing the data. In the example provided, we have a clean dataset, but that might not be always the case.

# # 3. Data Exploration

# - Look at the description of the dataset
# - Explore the dataset (what information can we gain from it and how can we use it to solve the defined problem)
# - Visualize our data to get more understanding on our dataset

# # 4. Evaluate our algorithms

# - We will import all the packages we need for our program to run in an end to end basis
# - Train different models and find out the best one
# - Get statistical Measures to evaluate our algorithm

# Import all the required packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# # Single Linear Regression: 

# # Description of project

# - We have a simple dataset where we have two variables: hours and scores. We have hours as independent variable and scores as dependent variable. 

# Use only one attribute to predict the outcome of the result
# - Predict the percentage of marks that a student is expected to score based upon number of hours studied

# - Question: What is the correation between number of hours studied and the expected score of the student?

# In[ ]:


score = pd.read_csv('../input/student_scores.csv')


# Exploring the dataset:

# In[ ]:


score.shape


# In[ ]:


score.head()


# Get the statistical details of the score dataset:

# In[ ]:


score.describe()


# Plot the dataset into a scatter plot diagram
# - Scatter plot gives us an idea about how two variabled corelate to each other. Since, we have only two variables in our dataset and one influences the other, scatter plot representation makes the most sense

# In[ ]:


score.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show() 


# - From the figure above we can conclude more the number of hours studied, higher the score

# Dividing the data into attributes and labels:
# - First column (Hours) is the attribute, which are independent variables 
# - Second column (Percentage) is the label, which are dependent variables

# In[ ]:


X = score.iloc[:, :-1].values  
y = score.iloc[:, 1].values  


# Splitting the data into training and test set:
# 80% training set and 20% test set

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# Choosing the Model:
# - We use linear regression to train a model because a single independent variable is used to predict the value of a dependent variable

# In[ ]:


model = LinearRegression()  
model.fit(X_train, y_train) 


# Since linear regression is basically:
# - y = mx + c, where m is slope and c is the intercept, we try to find the best possible values for the intercept (which is the goal)
# - Also, get the slope value

# In[ ]:


print(model.intercept_)  


# In[ ]:


print(model.coef_)  


# The coefficient is 9.91%. This means if a student studies one hour more than they previously studied for an exam, they can expect to achieve an increase of 9.91% in the score achieved by the student previously.

# Since, we have trained our model with the algorithm, the next step is to make predictions based upon the model on some input values

# In[ ]:


# y_pred contains all the predicted values for the X_test data
y_pred = model.predict(X_test)  


# In[ ]:


# Creating a dataframe to show how the prediction and actual data looks like
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df)


# Evaluate the Algorithm. We do this by using following methods:
# 1. Mean Absolute Error  
# 2. Mean Squared Error 
# 3. Root Mean Squared Error 

# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


# # Multiple Linear Regression

# ## Description of project:

# - The dataset includes four features: "Petrol Tax", "Average Income", "Paved Highways" and "Population Driver License Percent", which gives the the average "Petrol_Consumption" 
# - Goal: To find out consumption of petrol when provided 4 different featurs.

# In[ ]:


dataset = pd.read_csv("../input/petrol_consumption.csv")


# - The data was acquired from the web "https://drive.google.com/file/d/1mVmGNx6cbfvRHC_DvF12ZL3wGLSHD9f_/view"
# - Check some of the data and how the dataset looks like

# In[ ]:


dataset.head()


# To see statistical details of the dataset, we'll use the describe() command:

# In[ ]:


dataset.describe()


# Visualize the data using various plots: 
# 1. Box Plot
# 2. Histogram

# In[ ]:


dataset.plot(kind='box',layout=(2,2), sharex=False, sharey=False, figsize=(10,8))
plt.show()


# - Create histograms and describe the results

# In[ ]:


dataset.hist(figsize=(12,10))
plt.show()


# Prepare the Data:
# Use the first 4 columns as attributes and the final column as the label

# In[ ]:


X = dataset[['Petrol_tax', 'Average_income', 'Paved_Highways',  
       'Population_Driver_licence(%)']]
y = dataset['Petrol_Consumption']  


# - Split the dataset into the Training and Test Set (80/20)

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# # Models

# We create different models to see which works the best for the type of dataset we have:
# - Linear Regression
# - Gaussian Naive Bayes
# - Decision Tree Classifier
# - Stochastic Gradient Descent
# - K-Nearest Neighbor
# - Linear SVM

# We compare the accuracy of the different models

# - We use Linear Regression Method to train and fit the data

# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)  


# In case of multivariable linear regression, the regression model has to find the most optimal coefficients for all the attributes. To see what coefficients our regression model has chosen, execute the following script:

# In[ ]:


coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df


# Final step is to make prediction using the regressor model we created

# In[ ]:


y_pred = regressor.predict(X_test) 


# In[ ]:


df_regressor = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df_regressor)


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


# Another Algorithm we use here is "Gaussian Naive Bayes" which is a linear classifier so we can use it here.

# In[ ]:


model_gnb = GaussianNB()  
model_gnb.fit(X_train, y_train)
y_pred = model_gnb.predict(X_test) 


# In[ ]:


df_gnb = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df_gnb)


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


# Another Algorithm we use here is "Decision Tree Classifier" which is a linear classifier so we can use it here.

# In[ ]:


model_dtc = DecisionTreeClassifier() 
model_dtc.fit(X_train, y_train)
y_pred = model_dtc.predict(X_test) 


# In[ ]:


df_dtc = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df_dtc)


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


# Stochastic gradient descent is a popular algorithm for training a wide range of models in machine learning, including (linear) support vector machines, logistic regression (see, e.g., Vowpal Wabbit) and graphical models.

# In[ ]:


model_sgd = SGDClassifier()
model_sgd.fit(X_train, y_train)
y_pred = model_sgd.predict(X_test) 


# In[ ]:


df_sgd = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df_sgd)


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


# In[ ]:


model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
y_pred = model_knn.predict(X_test) 


# In[ ]:


df_knn = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df_knn)


# In[ ]:


model_svm = SVC(kernel="linear", C=0.025)
model_svm.fit(X_train, y_train)
y_pred = model_svm.predict(X_test) 


# In[ ]:


df_svm = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print(df_svm)

