#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Regression is one of the most important machine learning techniques. It is mainly used for identifying the relationship between two or more variables and for predicting a continuous variable. 
# 
# A particular dataset contains a response variable and a set of explanatory variables, where the explanatory variables are said to be the features of the response variable. Regression modeling is used to estimate the relationship between the variables and predict the value of response variable by finding a line of best fit that minimizes the Residual Sum of Squares (RSS) i.e. the difference between the actual value and predicted value. 
# 
# In this analysis, we will first build a simple and a multiple regression model using python library scikit learn and then use the Gradient Descent Algorithm, an optimization technique, to find the minimum of the function RSS. 
# 
# We must always remember that our results are based on the quantity and quality of the available dataset. 

# # Exploratory Data Analysis

# In[ ]:


# importing libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from math import sqrt


# In[ ]:


# Load Dataset
df_house = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')


# In[ ]:


# View the dataset
df_house.head()


# In[ ]:


# Total Rows and Columns
df_house.shape


# In[ ]:


# Checking Datatypes
df_house.dtypes


# #### Dropping Variable 

# There are some varibales such as ID and date which are irrelevant in predicting the value of the house. Therefore,before proceeding further we will drop these variables.

# In[ ]:


df_house = df_house.drop(['id','date'],axis =1)
df_house.head()


# ### Missing Values

# Before moving further with analysis it is important to check if there are any missing values in the dataset. As we can see from the code below, there are no missing values in the dataset. 

# In[ ]:


df_house.isnull().sum().sort_values(ascending = False)


# # Univariate Analysis

# To get an overview of the distribution of the response or dependent variable "price", we will plot a normal distribution curve which will tell us the skewness, and spread of the data. 

# In[ ]:


min_ = min(df_house['price'])
max_ = max(df_house['price'])
x = np.linspace(min_,max_,100)
mean = np.mean(df_house['price'])
std = np.std(df_house['price'])

# For Histogram
plt.hist(df_house['price'], bins=20, density=True, alpha=0.3, color='b')
y = norm.pdf(x,mean,std)

# For normal curve
plt.plot(x,y, color='red')


plt.show()


# As we can see from the above graph, that the variable 'price' is skewed towards the right. 

# In[ ]:


df_house['price'].describe()


# ### Boxplot

# The boxplot provides a visual summary of the following items in a data:
# 1. Inter Quartile Rande (IQR)
# 2. Median
# 3. Whiskers
# 4. Outliers

# In[ ]:


sns.boxplot(df_house['price'])


# The boxplot shows that there are some suspected outliers in the dataset. 

# # Correlation Matrix

# The correlation coefficient shows the relationship or association between the dependent variable and independent variable. The value of the coefficient is between -1 and 1. 
# 
# **Positive Correlation** - Positive Correlation lies between 0 and 1, where a value close to 1 represents a strong positive correlation and a value close to 0 represents a weak positive correlation. 
# 
# **Negative Correlation** - Negative Correlation lies between -1 and 0, where a value close to -1 represents a strong negative correlation and a value close to 0 represents a weak negative correlation. 
# 
# However, we must always remember that correlation does not imply causation.

# In[ ]:


correlation_matrix = df_house.corr()
print(correlation_matrix)


# In[ ]:


sns.heatmap(correlation_matrix)


# By looking at the value of correlation coefficients from above, we can see identify variables that have a strong (+ve or -ve) relationship with the variable 'price'. These are 'sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'bedrooms', 'sqft_basement'.

# # Multivariate Analysis

# In[ ]:


# Price & Sqft Living
df_house.plot(x='sqft_living',y='price',style = 'o')
plt.title('Sqft_Living Vs Price')


# In[ ]:


df_house.boxplot(column = ['price'],by='bedrooms')


# In[ ]:


df_house.plot(x='lat',y='price',style = 'o')
plt.title('lat Vs Price')


# # Simple Regression Model
# In a simple regression model, only a single explanatory variable is used to predict the value of response variable. In this case, the response variable is 'price' while explanatory variable would be 'sqft_living' as this variable had the highest positive correlation with price.

# ### Training and Test Dataset
# The dataset will be split into a training and test dataset. The training dataset will be used to fit the model i.e. it will find the estimated parameters of the model while the test dataset will be used for predicting the value of price. 

# In[ ]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df_house, test_size=0.20)


# In[ ]:


train.shape


# In[ ]:


test.shape


# ### Attributes and Labels

# In[ ]:


X_train_simple = train['sqft_living'].values.reshape(-1,1)
X_test_simple = test['sqft_living'].values.reshape(-1,1)

y_train_simple = train['price'].values.reshape(-1,1)
y_test_simple = test['price'].values.reshape(-1,1)


# ### Fitting A Simple Regression Model

# In[ ]:


model_s = LinearRegression()
model_s.fit(X_train_simple,y_train_simple)

print('Intercept: ', model_s.intercept_)

print('Sqft_living Coefficient: ', model_s.coef_)


# ### Predictions - Simple Regression

# In[ ]:


# Making Predictions
pred_simple = model_s.predict(X_test_simple)
pred_simple


# ### Residual Sum of Squares

# In[ ]:


# RSS
RSS_simple = np.sum((y_test_simple - pred_simple)**2)
print("RSS_simple: ", RSS_simple)


# ### Plotting line of Best Fit

# In[ ]:


plt.plot(test['sqft_living'],test['price'],'.',
        test['sqft_living'], pred_simple,'-')


# ### Residual Plot

# In[ ]:


sns.residplot('sqft_living','price', data = test, color = 'red')


# ### Covariance
# 

# In[ ]:


cov = pd.DataFrame.cov(df_house[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'bedrooms', 
                                 'sqft_basement','waterfront','floors']])
print(cov)


# In[ ]:


sns.heatmap(cov,fmt='g')
plt.show()


# From the above covariance matrix, we can see that the variables ('sqft_living','sqft_above', 'sqft_living15') have a strong relationship with one another.

# # Multiple Regression Model
# 
# We will build 3 multiple regression models by adding and removing different variables to see their affect on RSS. This will be done by using the built-in library in python i.e. scikit learn.
# 
# The training and testing datasets defined above will be used in this model.
# 
# ### Model 1
# In this model, we will take all the variables ('sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'bedrooms') which were highly corelated to the variable 'price' in our model. 
# 

# In[ ]:


# Separating Attributes and Labels
X_train = train[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'bedrooms']].values
X_test = test[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'bedrooms']].values
y_train = train['price'].values.reshape(-1,1)
y_test = test['price'].values.reshape(-1,1)


# In[ ]:


# Fitting Regression Model
model1 = LinearRegression()
model1.fit(X_train,y_train)

print('Intercept: ', model1.intercept_)

print('Coefficients: ', model1.coef_)

df1 = pd.DataFrame(model1.coef_, columns = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view',
                                            'bedrooms'])
print(df1)


# The value of the coefficients, for e.g. sqft_living, represents the predicted change in the value of price per unit change in the value of square feet.

# In[ ]:


# Making the predictions od model 1
pred1 = model1.predict(X_test)
pred1


# In[ ]:


# RSS of model 1
RSS_1 = np.sum((y_test - pred1)**2)
print("RSS_1: ", RSS_1)


# ### Model 2
# In model 2 we will add the variables  ('sqft_basement', 'lat') as they are correlated with "price".

# In[ ]:


# Separating Attributes and Labels
X_train = train[['sqft_living', 'grade', 'bathrooms', 'view', 'bedrooms', 'sqft_above', 'sqft_living15','sqft_basement'
                 ,'lat']].values
X_test = test[['sqft_living', 'grade', 'bathrooms', 'view', 'bedrooms','sqft_above', 'sqft_living15','sqft_basement'
              ,'lat']].values

y_train = train['price'].values.reshape(-1,1)
y_test = test['price'].values.reshape(-1,1)


# In[ ]:


# Fitting a Regression Model 2
model2 = LinearRegression()
model2.fit(X_train,y_train)

print('Intercept: ', model2.intercept_)

print('Coefficients: ', model2.coef_)

df2 = pd.DataFrame(model2.coef_, columns = ['sqft_living', 'grade', 'bathrooms', 'view', 'bedrooms', 
                                           'sqft_above', 'sqft_living15','sqft_basement','lat'])
print(df2)


# In[ ]:


# Making Predictions of Model 2
pred2 = model2.predict(X_test)
pred2


# In[ ]:


# RSS of model 2
RSS_2 = np.sum((y_test - pred2)**2)
print("RSS_2: ", np.sum((y_test - pred2)**2))


# ### Model 3
# In this model, we will remove the variables ('sqft_above', 'sqft_living15','sqft_basement') as they are strongly related to the variable sqft_living.
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


# Separating Attributes and Labels
X_train = train[['sqft_living', 'grade', 'bathrooms', 'bedrooms','view','lat']].values
X_test = test[['sqft_living', 'grade', 'bathrooms', 'bedrooms','view','lat']].values


# In[ ]:


# Fiting a Regression Model
model3 = LinearRegression()
model3.fit(X_train,y_train)

print('Intercept: ', model3.intercept_)

print('Coefficients: ', model3.coef_)

df3 = pd.DataFrame(model3.coef_, columns = ['sqft_living', 'grade', 'bathrooms', 'bedrooms', 'view','lat'])
print(df3)


# In[ ]:


# Predicting the value
pred3 = model3.predict(X_test)
pred3


# In[ ]:


# RSS
RSS_3 = np.sum((y_test - pred3)**2)
print("RSS_3: ", np.sum((y_test - pred3)**2))


# As we can see, adding or removing variables may result in the Residual Sum of Squares to rise.

# In[ ]:


RSS = pd.DataFrame(np.array([[RSS_1,RSS_2,RSS_3]]),columns = ['RSS_1','RSS_2','RSS_3'])
print(RSS)


# # Gradient Descent
# 
# In a simple regression model, we try to estimate the value of the response variable (house price) from a single explanatory varibale which in this case is "sqft_living" by fitting a line that best fits our model. 
# 
# But how do we measure the quality or performance of our model? 
# 
# This is done by defining a cost function such as Residual Sum of Squares (RSS) in terms of our estimated parameters (w0, w1) where w0 is the intercept and w1 is the coefficeint of 'sqft_living'. Therefore, our main goal in fitting a model is to minimize cost function, RSS(w0,w1),  over all possible values of (w0, w1) i.e. search over the space of all possible lines and find the line that minimize the RSS.
# 
# Gradient Descent is an optimization technique in Machine Learning that allow us to find specific values of w0 and w1 which minimizes our RSS. 
# 
# To understand this in more detail, we can try to find the minimum and maximum of the cost function analyticaly. For example, in order to find the minimum of a convex function, we will compute the derivative of that function and equate it to 0. However, this could become computationaly intensive if there are more than 1 variables. 
# 
# 

# A better approach would then be to use hill climbing or descent technique for finding the maximum or minimum respectively. Instead of equating the derivative to 0, we move along the curve from one point to another by updating the value or vector of the estimated parameter W (or weights of the given input features). This is done through an iterative process by defining a stepsize and convergence criteria. 
# 
# First, we assume an initial value of our estimated parameters (w,regression coefficients) and compute the derivative of the cost function (RSS) at that point. Then at each iteration the previous value of w is either increased or decreased by the amount based on the derivative as determined by the stepsize (Instead of the derivative, the value is increased or decresed by the amount of stepsize). 
# 
# In case of min, if the value of the derivative is -ve, increase the value of w. If the value of the derivative is +ve, decrease the value of w.
# 
# Laslty, how do we assess the convergence? The algorithm will not converge until the magnitude of the derivative is less than the tolerance level i.e. threshold 'e', which will be a very small number. The threshold will be set by us and the algorithm will not terminate until the condition is satisfied.
# 

# ## Simple Regression - Gradient Descent
# To understand Gradient Descent, let's start with simple regression by taking a single input feature 'sqft_living' from the housing dataset.
# 
# The following function takes data, input features, output and returns a feature_matrix which will consist a first column of ones and then the value of input features in the order defined. It also returns an output 'price' of the dataset. 
# 
# 
# ### Input and Output

# In[ ]:


def get_data(data,features, output):
    data['constant'] = 1
    features = ['constant'] + features

    features_new = data[features]
    feature_matrix = np.asarray(features_new)
    
    output_data = data[output]
    output_array = output_data.to_numpy()
    return(feature_matrix,output_array)


# ### Predictions
# 
# Then, we get the 1D array predictions by multiplying 2D rfeature_matrix with 1D regression weights which is a dot product between the two vectors. We will define the initial weights to be [-47000, 1]

# In[ ]:


def prediction(feature_matrix, weights):
    predictions = np.dot(feature_matrix,weights)
    return(predictions)


# ### Derivative
# 
# Now we take the derivative of the cost function. A derivative function is then defined which takes feature and error array and returns a number. 

# In[ ]:


def feature_derivative(errors,feature):
    derivative = 2*(np.dot(errors,feature))
    return(derivative)


# ### Gradient Descent Algorithm
# 

# In[ ]:


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        predictions = prediction(feature_matrix,weights)
        
        errors = predictions - output
        
        gradient_sum_squares = 0
        for i in range(len(weights)):
            derivative = feature_derivative(errors,feature_matrix[:,i])
            
            gradient_sum_squares = derivative**2 + gradient_sum_squares
            
            # subtract the step size times the derivative from the current weight
            weights = weights - step_size*(derivative)
            
            gradient_magnitude = sqrt(gradient_sum_squares)
            if gradient_magnitude < tolerance:
                converged = True
    return(weights)  


# ### Define Parameters
# 
# 1. features
# 2. output
# 3. errors
# 4. tolerance
# 5. stepsize

# In[ ]:


(feature_matrix,output_array) = get_data(train, ['sqft_living'],'price')
initial_weights = np.array([-47000., 1.])
predictions = prediction(feature_matrix, initial_weights)
errors = output_array - predictions
step_size = 7e-12
tolerance = 2.5e7


# In[ ]:


weights = regression_gradient_descent(feature_matrix, output_array, initial_weights, step_size, tolerance)


# In[ ]:


print(weights)


# ### Residual Sum of Squares

# In[ ]:


# Using test data to compute RSS
(feature_matrix_test,output_array_test) = get_data(test, ['sqft_living'],'price')


# In[ ]:


prediction_test = prediction(feature_matrix_test, weights)
errors = output_array_test - prediction_test
RSS_simple_GD = np.sum((errors)**2)
print("RSS_simple_GD: ", np.sum((errors)**2))


# In[ ]:


print("RSS using Simple Regression: ", RSS_simple)


# From above we can see that the RSS of Simple Regression using the optimization technique, Gradient Descent, is much lower than RSS of simple regression using a pyhton library.

# In[ ]:




