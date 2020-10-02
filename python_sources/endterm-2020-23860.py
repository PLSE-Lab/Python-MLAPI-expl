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

import matplotlib.pyplot as plt     # for visualization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


# use Pandas to read in csv files. The pd.read_csv() method creates a DataFrame from a csv file

# In[ ]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# check out the size of the data

# In[ ]:


print("Train data shape:", train.shape)
print("Test data shape:", test.shape)


# look at a few rows using the DataFrame.head() method

# In[ ]:


print(train.head())


# to do some plotting

# In[ ]:


plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)


# to get more information like count, mean, std, min, max etc

# In[ ]:


print (train.SalePrice.describe())


# to plot a histogram of SalePrice

# In[ ]:


print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()


# use np.log() to transform train.SalePric and calculate the skewness a second time, as well as re-plot the data

# In[ ]:


target = np.log(train.SalePrice)
print ("\n Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()


# return a subset of columns matching the specified data types

# In[ ]:


numeric_features = train.select_dtypes(include=[np.number])
print(numeric_features.dtypes)


# displays the correlation between the columns and examine the correlations between the features and the target.

# In[ ]:


corr = numeric_features.corr()


# The first five features are the most positively correlated with SalePrice, while the next five are the most negatively correlated.

# In[ ]:


print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])


# to get the unique values that a particular column has.

# In[ ]:


print(train.OverallQual.unique())


# investigate the relationship between OverallQual and SalePrice.
# We set index='OverallQual' and values='SalePrice'. We chose to look at the median here.

# In[ ]:


quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
print(quality_pivot)


# visualize this pivot table more easily, we can create a bar plot. Notice that the median sales price strictly increases as Overall Quality increases.

# In[ ]:


quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# to generate some scatter plots and visualize the relationship between the Ground Living Area(GrLivArea) and SalePrice

# In[ ]:


plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()


# do the same for GarageArea.

# In[ ]:


plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


# create a new dataframe with some outliers removed

# In[ ]:


train = train[train['GarageArea'] < 1200]


# display the previous graph again without outliers

# In[ ]:


plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600)     # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


# create a DataFrame to view the top null columns and return the counts of the null values in each column

# In[ ]:


nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)


# to return a list of the unique values

# In[ ]:


print ("Unique values are:", train.MiscFeature.unique())


# consider the non-numeric features and display details of columns

# In[ ]:


categoricals = train.select_dtypes(exclude=[np.number])
print(categoricals.describe())


# When transforming features, it's important to remember that any transformations that you've applied to the training data before fitting the model must be applied to the test data.

# In[ ]:


print ("Original: \n")
print (train.Street.value_counts(), "\n")


# our model needs numerical data, so we will use one-hot encoding to transform the data into a Boolean column. Create a new column called enc_street. The pd.get_dummies() method will handle this for us

# In[ ]:


train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(test.Street, drop_first=True)


# Pave and Grvl values converted into 1 and 0

# In[ ]:


print ('Encoded: \n')
print (train.enc_street.value_counts())


# look at SaleCondition by constructing and plotting a pivot table, as we did above for OverallQual

# In[ ]:


condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# encode this SaleCondition as a new feature by using a similar method that we used for Street above

# In[ ]:


def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)


# explore this newly modified feature as a plot.

# In[ ]:


condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# Dealing with missing values. We'll fill the missing values with an average value and then assign the results to data. This is a method of interpolation  

# In[ ]:


data = train.select_dtypes(include=[np.number]).interpolate().dropna()


# Check if the all of the columns have 0 null values.

# In[ ]:


print(sum(data.isnull().sum() != 0))


# separate the features and the target variable for modeling. We will assign the features to X and the target variable(Sales Price)to y.

# In[ ]:


y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)


# In this example, about 33% of the data is devoted to the hold-out set.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)


#  # Linear Regression Model 

# In[ ]:


lr = linear_model.LinearRegression()


# lr.fit() method will fit the linear regression on the features and target variable that we pass.

# In[ ]:


model = lr.fit(X_train, y_train)


# Evaluate the performance and visualize results

# In[ ]:


print("R^2 is: \n", model.score(X_test, y_test))


# use the model we have built to make predictions on the test data set.

# In[ ]:


predictions = model.predict(X_test)


# calculates the rmse

# In[ ]:


print('RMSE is: \n', mean_squared_error(y_test, predictions))


# view this relationship between predictions and actual_values graphically with a scatter plot.

# In[ ]:


actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') 
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()


# try using Ridge Regularization to decrease the influence of less important features 

# In[ ]:


for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()


# In[ ]:


print("R^2 is: \n", model.score(X_test, y_test))


#  create a csv that contains the predicted SalePrice for each observation in the test.csv dataset.

# In[ ]:


submission = pd.DataFrame()


# The first column must the contain the ID from the test data.

# In[ ]:


submission['Id'] = test.Id


# select the features from the test data for the model as we did above.

# In[ ]:


feats = test.select_dtypes(
    include=[np.number]).drop(['Id'], axis=1).interpolate()


# generate predictions

# In[ ]:


predictions = model.predict(feats)


# transform the predictions to the correct form. apply np.exp() to our predictions becasuse we have taken the logarithm(np.log()) previously.

# In[ ]:


final_predictions = np.exp(predictions)


# check the difference

# In[ ]:


print("Original predictions are: \n", predictions[:10], "\n")
print("Final predictions are: \n", final_predictions[:10])


# assign these predictions and check

# In[ ]:


submission['SalePrice'] = final_predictions
print(submission.head())


# export to a .csv file as Kaggle expects.

# In[ ]:


submission.to_csv('submission1.csv', index=False)

