#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd  # To read data
import numpy as np # To calculate data


# ### Less is more. This is my approach to everything, and data science projects are no exception. I like clean, simple yet meaningful insights, which are straight to the point. 

# # 1.0 Importing the Data
# 
# ### Load the csv file from directory to the dataframe
#  
# - Use the .head() function to catch a snapshot of the top end of the dataframe

# In[ ]:


df = pd.read_csv("/kaggle/input/house-prices-dataset/train.csv")
df_test = pd.read_csv("/kaggle/input/house-prices-dataset/test.csv")
df.head()


# - #### Display the dataframe statistical summary for better overview of thats low or high. Using the function - .describe()
# 

# In[ ]:


df.describe()


# - Lets check the data types for each column.

# In[ ]:


df.dtypes


# # 2.0 Data Cleaning

# #### Lets take a quick look at a correlation heatmap/matrix and determine the most relevant data to work with, instead of just going through the entire 81 column list, looking which columns to drop.

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 7,7 
import seaborn as sns
import numpy as np
sns.set(color_codes=True, font_scale=1.2)

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


import heatmapk
from heatmapk import heatmap, corrplot
plt.figure(figsize=(10, 10))
corrplot(df.corr(), size_scale=300);


# - We can determine from this heatmap, the biggest variables that the Sale Prices is corelated to, which are OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF and 1stFlrSF. Atleast from what i cant make out real quick. 
# - Lets confirm the exact correlation coeficient between these variables and our "SalePrice" value. Lets narrow down the exact variables via their coeficient.

# In[ ]:


corr_matrix=df.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)


# - Lets select a new dataframe just with these main above 60%ish correlation having percentile variables, so it will be easier to process.

# In[ ]:


df1 = df[['OverallQual', 'GrLivArea', 'GarageCars', 
          'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'SalePrice']].copy()
df1.head()


# Much easier on the eyes.
# - Now lets find out if and how many missing values we are dealing with, within this new data frame AKA our features list. 
# - We will do that by applying the - .isnull() method.

# In[ ]:


df1.isnull().sum()


# - Good stuff. No need to drop or insert any means within any of our columns since there are no missing values. Lets move on.

# # 3.0 Explolatory Data Anlysis

# - Lets put up some histograms to observe any crazy skewsness or abnormalities from our variables within an eyes view.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df1.hist(bins=50, figsize=(20,15))
plt.savefig("attribute_histogram_plots")
plt.show()


# - All seems to be doing fine so far. Lets use seaborn regression functions to plot our Linear Regression of each attribute in comparison to our Target value - Sale Price

# In[ ]:



sns.regplot(x='1stFlrSF', y='SalePrice', data=df1)


# In[ ]:


df1[['1stFlrSF', 'SalePrice']].corr()


# In[ ]:


sns.regplot(x='GarageArea', y='SalePrice', data=df1)


# In[ ]:


df1[['GarageArea','SalePrice']].corr()


# In[ ]:


sns.regplot(x='GarageCars', y='SalePrice', data=df1)


# In[ ]:


df1[['GarageCars', 'SalePrice']].corr()


# In[ ]:


sns.regplot(x='GrLivArea', y='SalePrice', data=df1)


# In[ ]:


df1[['GrLivArea', 'SalePrice']].corr()


# In[ ]:


sns.regplot(x='OverallQual', y='SalePrice', data=df1)


# In[ ]:


df1[['OverallQual', 'SalePrice']].corr()


# In[ ]:


sns.regplot(x='TotalBsmtSF', y='SalePrice', data=df1)


# In[ ]:


df1[['TotalBsmtSF', 'SalePrice']].corr()


# ### Lets calculate the P-value. It is the probability value between two variables is statistically significant. For example:
# - if we choose significance level of 0.05, that means that we are 95% confident that the correlation between the variables is significant.

# By convention, when the:
# 
# - p-value is  < 0.001: we say there is strong evidence that the correlation is significant.
# - the p-value is < 0.05: there is moderate evidence that the correlation is significant.
# - the p-value is  < 0.1: there is weak evidence that the correlation is significant.
# - the p-value is  > 0.1: there is no evidence that the correlation is significant.

# In[ ]:


from scipy import stats


# - ### Lets calculate the Pearson Correlation Coefficient and P-value of OverallQual and SalePrice:

# In[ ]:


pearson_coef, p_value=stats.pearsonr(df1['OverallQual'], df1['SalePrice'])
print('The Pearson Correlation Coefficient is ', pearson_coef, 'with a P-value of P =', p_value)


# #### Conclusion:
# 
# - Since the p-value is < 0.001, the correlation between OverallQual and SalePrice is statistically _SIGNIFICANT_, and the coefficient of ~ 0.79 shows that the relationship is _QUITE STRONG_.

# - ### Lets calculate the Pearson Correlation Coefficient and P-value of GrLivAea and SalePrice:

# In[ ]:


pearson_coef, p_value=stats.pearsonr(df1['GrLivArea'], df1['SalePrice'])
print('The Pearson Correlation Coefficient is ', pearson_coef, 'with a P-value of P=', p_value)


# #### Conclusion:
# 
# - Since the p-value is < 0.001, the correlation between GrLivArea and SalePrice is statistically _SIGNIFICANT_, and the coefficient of ~ 0.70 shows that the relationship is _MODERATELY STRONG_.

# - ### Lets calculate the Pearson Correlation Coefficient and P-value of GarageCars and SalePrice:

# In[ ]:


pearson_coef, p_value=stats.pearsonr(df1['GarageCars'], df1['SalePrice'])
print('The Pearson Correlation Coefficient is ', pearson_coef, 'with a P-value of P=', p_value)


# #### Conclusion:
# 
# - Since the p-value is < 0.001, the correlation between GarageCars and SalePrice is statistically _SIGNIFICANT_, and the coefficient of ~ 0.64 shows that the relationship is _MODERATELY STRONG_.

# - ### Lets calculate the Pearson Correlation Coefficient and P-value of GarageArea and SalePrice:

# In[ ]:


pearson_coef, p_value=stats.pearsonr(df1['GarageArea'], df1['SalePrice'])
print('The Pearson Correlation Coefficient is ', pearson_coef, 'with a P-value of P=', p_value)


# #### Conclusion:
# 
# - Since the p-value is < 0.001, the correlation between GarageArea and SalePrice is statistically _SIGNIFICANT_, and the coefficient of ~ 0.62 shows that the relationship is _MODERATELY STRONG_.

# - ### Lets calculate the Pearson Correlation Coefficient and P-value of TotalBsmtSF and SalePrice:

# In[ ]:


pearson_coef, p_value=stats.pearsonr(df1['TotalBsmtSF'], df1['SalePrice'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P=', p_value)


# #### Conclusion:
# 
# - Since the p-value is < 0.001, the correlation between TotalBsmtSF and SalePrice is statistically _SIGNIFICANT_, and the coefficient of ~ 0.61 shows that the relationship is _MODERATE_.

# - ### Lets calculate the Pearson Correlation Coefficient and P-value of 1StFlrSF and SalePrice:

# In[ ]:


pearson_coef, p_value=stats.pearsonr(df1['1stFlrSF'], df1['SalePrice'])
print('The Pearson Correlation Coefficient is', pearson_coef, 'with a P-value of P=', p_value)


# #### Conclusion:
# 
# - Since the p-value is < 0.001, the correlation between 1stFLrSF and SalePrice is statistically _SIGNIFICANT_, and the coefficient of ~ 0.60 shows that the relationship is _MODERATE_.

#    - ### I  summary, we came to a conclusion what our data looks like and which variables are important to take into account when predicting the house price. We have narrowed it down to the following 6 variables:
#  - OverallQual
#  - GrLivArea
#  - GarageCars
#  - GarageArea
#  - TotalBsmtSF
#  - 1stFlrSF
# 
# 
# Out of all of them, variable  _'OverallQual'_ has the Strongest positive and Significant relationship to our target - 'SalePrice'.
# 
# - ### As we will be moving into building our machine learning models to automate our analysis, feeding the model with variables we selected in conclusion, which meaningfully affect our target variable will improve our model's prediction performance.

# # 4.0 Model Development

# #### In this part of the project, we will develop several models that will predict the price of the house using the variables or features above. This is just an estimate obviously, but should give us an objective idea of how much the house should cost.

# ### Setup
# 
# - Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


# ### 1. Linear Regression and Multiple Linear Regression

# - In our first example of Data Model we will be using is _Simple Linear Regression_
# 
# Simple Linear Regression is a method to help us understand the relationship between two variables:
# - The predictor/independent variable(X) - OverallQual
# - The response/dependent variable (that we want to predict), (Y) - SalePrice

# ### Creating train and test dataset
# 
# - Lets split the dataset into train and test sets, 80% of the entire data for training and the 20% for testing. LEts create a mask to select random rows using np.random.rand() function:
# 

# In[ ]:


msk = np.random.rand(len(df)) < 0.8
train =df1[msk]
test =df1[~msk]


# - Train data distribution

# In[ ]:


plt.scatter(train.OverallQual, train.SalePrice, color='green')
plt.xlabel('Overall Quality')
plt.ylabel('Sales Price')
plt.show()


# ### Modeling

# In[ ]:


from sklearn import linear_model

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['OverallQual']])
train_y = np.asanyarray(train[['SalePrice']])
regr.fit (train_x, train_y)

# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)


# ### Plot outputs

# In[ ]:


plt.scatter(train.OverallQual,  train.SalePrice, color ='green')
plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
plt.xlabel('Overall Quality')
plt.ylabel('Sales Price')


# ### Evaluation

# In[ ]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['OverallQual']])
test_y = np.asanyarray(test[['SalePrice']])
test_y_hat = regr.predict(test_x)

print('Mean Absolute Error: %.2f' % np.mean(np.absolute(test_y_hat-test_y)))
print('Residual sum of squares (MSE): %.2f' % np.mean((test_y_hat - test_y) **2))
print('R2-score: of %.2f' % r2_score(test_y_hat, test_y))     


# #### - Lets fit the model with the feature 'OverallQual'

# In[ ]:


X=df1[['OverallQual']]
Y=df1['SalePrice']
lm=LinearRegression()
lm
lm.fit(X, Y)


# Output a prediction:

# In[ ]:


Yhat=lm.predict(X)
Yhat[0:5]


# - What is the value of intercept (a)?

# In[ ]:


lm.intercept_


# - What is the value of the Slope (b)?

# In[ ]:


lm.coef_


# ### What is the final estimated linear model we get?
# 
# - Yhat = a +bX
# 
# And putting in our actual values we get:
# 
# Price = -96206 + 45435 * OverallQual

# - Convert the output to a readable Data Frame:

# In[ ]:


Yhat_df1 = pd.DataFrame(Yhat)
Yhat_df1.columns = ['SalePrice']
Yhat_df1.head(10)


# - Add 'ID' Column:

# In[ ]:


Id = df_test[['Id']]
Yhat = pd.concat([Id,Yhat_df1],axis=1)
Yhat.head(10)


# ### Multiple Linear Regression

# - #### From the previous analysis, we know that we have 6 good predictor variables. Lets use them to create a multiple linear regression model.

# - Lets fit linear regression model to predict 'SalePrice' using list of the main features we analyzed earlier.
#  - Lets convert this list of features to a 'list' right away.

# In[ ]:


Z = df1[['OverallQual', 'GrLivArea', 'GarageCars', 
          'GarageArea', 'TotalBsmtSF', '1stFlrSF']]


# - Fit the linear model using the above listed variables.

# In[ ]:


lm.fit(Z, df1['SalePrice'])


# - Value of INTERCEPT (a)?

# In[ ]:


lm.intercept_


# - Values of the coefficients (b1,b2,b3,b4,b5,b6)?

# In[ ]:


lm.coef_


# - What's the stimated linear model we get?
# 
# Linear function structure as follows:
# 
#     Yhat = a +b1X1 + b2X2 + b3X3 + b4X4 + b5X5 + b6X6
#     
# - Price = -102650 + 2.39970394e+04*OverallQual +4.31228864e+01*GrLiveArea + 1.45151932e+04*GarageCars +1.56639341e+01*GarageArea + 2.43907676e+01* TotalBsmtSF + 1.11859135e+01* 1stFlrSF
#        
#        

# - Fit the model with the features data

# In[ ]:


y_output = lm.predict(Z)


# - Convert the output to a readable Data Frame:

# In[ ]:


y_output_df1 = pd.DataFrame(y_output)
y_output_df1.columns = ['SalePrice']
y_output_df1.head(10)


# - Add 'ID' Column:
# 

# In[ ]:


Id = df_test[['Id']]
Y_output = pd.concat([Id,y_output_df1],axis=1)
Y_output.head(10)


# - Now lets calculate the R^2 using the features compared to the 'SalePrice'

# In[ ]:


x2 = Z
y2 = df1['SalePrice']
lm.fit(x2, y2)
lm.score(x2,y2)


# #### Create a list of tuples: first element in the tuple contains the name of the estimator:
# - 'scale', 'polynomial', 'model'
# 
# #### The second element contains the model constructor
# - StandardScaler()
# - PolynomialFeatures(include_bias=False)
# - LinearRegression()

# In[ ]:


Input=[('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)),
      ('model', LinearRegression())]


# #### Use the list to create a pipeline object, predit the'SalePrice', fit the object using features in the features list, then fit the model and calculare the R^2

# ## To be fair, all of this seems like a lot of work for simple prediction of house prices, and there should be easier more efficient way to calculate this. Lets check out XGboost.

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import numpy as np


test = pd.read_csv("/kaggle/input/house-prices-dataset/test.csv")
df.head()
#Opening our file with the training data in
train = pd.read_csv("/kaggle/input/house-prices-dataset/train.csv")

#We are trying to predict the sale price column
target = train.SalePrice

#Get rid of the answer and anything thats not an object
train = train.drop(['SalePrice'],axis=1).select_dtypes(exclude=['object'])

#Split the data into test and validation
train_X, test_X, train_y, test_y = train_test_split(train,target,test_size=0.25)

#Impute all the NaNs
my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.fit_transform(test_X)

#Simplist XGBRegressor
#my_model = XGBRegressor()
#my_model.fit(train_X, train_y)

my_model = XGBRegressor(n_estimators=300, learning_rate=0.08)
my_model.fit(train_X, train_y, early_stopping_rounds=4, 
             eval_set=[(test_X, test_y)], verbose=False)


#Make predictions
predictions = my_model.predict(test_X)

print("Mean absolute error = " + str(mean_absolute_error(predictions,test_y)))


# In[ ]:





# - Similary evaluate a model and make predictions just as we would do in scikit-learn

# In[ ]:


# make predictions

predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print('Mean Absolute Error : ' + str(mean_absolute_error(predictions, test_y)))


# 

# # Conclusion

# ### After all the hard work, in the beginning, since it is my first ever Machine Learning excercise, i came across XGBoost. This should be prerequisite for every beginner. This not only shortened the time, was efficient, easier on the eyes, but also powerful and more accurate predictions. 

# In[ ]:




