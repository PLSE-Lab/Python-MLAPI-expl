#!/usr/bin/env python
# coding: utf-8

# **Prefix**
# 
# Hi, everyone! As a high school student interested in computer science and economics, this is my first taste of applying computational tools and machine learning algorithms, after learning Python programming and some machine learning.
# 
# I want to thank kagglers Junying(Emma) Zhang, Vijay Gupta, and meikegw whose posts provide valuable guidance to me, since this is my first time to work on this platform. 
# 
# Sharing my approach here, I hope it could, in turn, help other beginners to start out, and I am also excited to learn and improve, so please comment anything you think I should work more on.
# 
# I would also continue to improve my work, so check back for any update!
# 
# Thank you!

# **Keywords**
# 
# - NaN Value Filling
# - Deleting Outliers
# - Log Transformation
# - Correlation Analysis
# - Encoding Categorial Variables
# - Ridge Regression
# - Cross Validation

# **Detailed Approach**

# ***Importing packages***
# 
# The first step is to import the packages that I will use next, such as matplotlib for ploting, numpy for computing, and seaborn for customizing plot style.

# In[ ]:


# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# use seaborn to costomize matplotlib plot theme
import seaborn as sns
sns.set_style("darkgrid")
import scipy.stats as stats
# use ridge regession library provided by sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# In[ ]:


# filter unnecessary warnings
# thanks to Ahmad Javed for teaching me this method!
import warnings 
warnings.filterwarnings('ignore')


# ***Import data for this competition***

# In[ ]:


# load data
dataTrain=pd.read_csv('../input/train.csv')
dataTest=pd.read_csv('../input/test.csv')


# ***Familiarizing ourselves with the data***
# 
# By printing out a portion of the data set and the dimensions (shape) of it, I could know that the training set is fairly large with 1460 entries, and there are 80 features provided.

# In[ ]:


print(dataTrain.head(), '\n', dataTrain.shape)
print(100 * '*')
print(dataTest.head(), '\n', dataTest.shape)


# ***Deleting outliers***
# 
# Outliers could sometimes have great influence on statistical results, therefore, when building models, we should delete outliers. Here, I plot the value of the response variable, SalePrice, against value of one feature, GrLivArea. From the scatter plot, I notice that the two points at the lower right corner are clearly outliers, so I drop these two data entries from the training set.

# In[ ]:


# check possible outliers in the dependent variable, SalePrice
plt.scatter(x='GrLivArea', y='SalePrice', data=dataTrain, color='b', marker='*')
plt.show()


# In[ ]:


print('before removal: ', dataTrain.shape)
# remove two outliers
dataTrain.drop(dataTrain[(dataTrain['GrLivArea']>4000) & (dataTrain['SalePrice']<200000)].index, inplace=True)
print('after removal: ', dataTrain.shape)


# ***Filling NaN values***
# 
# Data set containing NaN values couldn't be directly modeled. So I first check the number of NaN values contained by column. I notice that a lot of categorial features contain high number of NaN values. Checking the description of the data set, I find that it is because some values called 'NA', for example meaning that the house doesn't have a fireplace, are incorrectly regarded as missing values by panda library during data importing. Therefore, I need to fill those values with their orininal true values. 
# 
# However, there are some data points that are missing, so for categorial values that are missing, I fill the mode, and for numerical values, I fill the median. The use of median instead of the mean is to avoid possible influence of skewness. 

# In[ ]:


# check NaN values in the training and test set
print(dataTrain.isnull().sum().sort_values(ascending=False).head(25))
print(100*'*')
print(dataTest.isnull().sum().sort_values(ascending=False).head(35))


# In[ ]:


# record all train columns
dicColumn = dataTrain.columns
for nameColumn in dicColumn:
    # object features
    if dataTrain[nameColumn].dtype == 'object':
        # value that should be 'NA', meaning 'no'
        if dataTrain[nameColumn].isnull().sum()>30:
            dataTrain[nameColumn].fillna(value='NA', inplace=True)
        # value that is unavailable
        else:
            dataTrain[nameColumn].fillna(value=dataTrain[nameColumn].mode()[0], inplace=True)
    # numerical features
    else:
        if dataTrain[nameColumn].isnull().any():
            # fill with median
            dataTrain[nameColumn].fillna(value=dataTrain[nameColumn].median(), inplace=True)
            
# record all test columns
dicColumnT = dataTest.columns
for nameColumn in dicColumnT:
    # object features
    if dataTest[nameColumn].dtype == 'object':
        # value that should be 'NA', meaning 'no'
        if dataTest[nameColumn].isnull().sum()>30:
            dataTest[nameColumn].fillna(value='NA', inplace=True)
        # value that is unavailable
        else:
            dataTest[nameColumn].fillna(value=dataTest[nameColumn].mode()[0], inplace=True)
    # numerical features
    else:
        if dataTest[nameColumn].isnull().any():
            # fill with median
            dataTest[nameColumn].fillna(value=dataTest[nameColumn].median(), inplace=True)
            
# check NaN values in the training and test set
print(dataTrain.isnull().sum().sort_values(ascending=False).head(10))
print(100*'*')
print(dataTest.isnull().sum().sort_values(ascending=False).head(10))


# ***Correlation analysis***
# 
# I first plot the correlation matrix in heat map format. Looking at the map, I find that there are four pairs of  features that are highly correlated. Because in regression, features should be as independent from each other as possible and correlation among features could bring undesirable results, in case of a pair of correlated features, I will only keep the feature that has higher correlation to the dependent variable, SalePrice.
# 
# After that, I calculate the correlation factors between features and SalePrice and sort them in ascending order. Some correlation factors are too small that they may be caused purely by chance, therefore, I delete features that have correlation factors less than 0.3.

# In[ ]:


plt.figure(figsize=(15,12))
sns.heatmap(dataTrain.corr(), vmax=0.9)


# In[ ]:


# correlation between features
print(dataTrain.corr()['YearBuilt']['GarageYrBlt'])
# correlation between features and saleprice
print(dataTrain.corr()['SalePrice']['YearBuilt'])
print(dataTrain.corr()['SalePrice']['GarageYrBlt'])


# In[ ]:


# correlation between features
print(dataTrain.corr()['GarageArea']['GarageCars'])
# correlation between features and saleprice
print(dataTrain.corr()['SalePrice']['GarageArea'])
print(dataTrain.corr()['SalePrice']['GarageCars'])


# In[ ]:


# correlation between features
print(dataTrain.corr()['TotalBsmtSF']['1stFlrSF'])
# correlation between features and saleprice
print(dataTrain.corr()['SalePrice']['TotalBsmtSF'])
print(dataTrain.corr()['SalePrice']['1stFlrSF'])


# In[ ]:


# correlation between features
print(dataTrain.corr()['GrLivArea']['TotRmsAbvGrd'])
# correlation between features and saleprice
print(dataTrain.corr()['SalePrice']['GrLivArea'])
print(dataTrain.corr()['SalePrice']['TotRmsAbvGrd'])


# In[ ]:


# correlation between features
print(dataTrain.corr()['GrLivArea']['TotRmsAbvGrd'])
# correlation between features and saleprice
print(dataTrain.corr()['SalePrice']['GrLivArea'])
print(dataTrain.corr()['SalePrice']['TotRmsAbvGrd'])


# In[ ]:


print(dataTrain.shape, dataTest.shape)
dataTrain.drop(['GarageYrBlt','GarageArea','1stFlrSF','TotRmsAbvGrd'], axis=1, inplace=True)
dataTest.drop(['GarageYrBlt','GarageArea','1stFlrSF','TotRmsAbvGrd'], axis=1, inplace=True)
print(dataTrain.shape, dataTest.shape)


# In[ ]:


# correlation between SalePrice and features in ascending order
print(dataTrain.corr()['SalePrice'].abs().sort_values(ascending=True).head(25))


# In[ ]:


print(dataTrain.shape, dataTest.shape)
# delete features w=with correlation factors less than 0.3
irrelatedCol = dataTrain.corr()['SalePrice'].abs().sort_values(ascending=True).head(19).index
dataTrain.drop(irrelatedCol, axis=1, inplace=True)
dataTest.drop(irrelatedCol, axis=1, inplace=True)
print(dataTrain.shape, dataTest.shape)


# ***Log transformation***
# 
# Plotting the dependent variable, SalePrice, shows that it is actually skewed. Therefore, a log transformation is needed to yield more accurate result. Due to log transformation, the predicted results need a step more to transform back to the original scale.

# In[ ]:


# check skewness in dependent variable
sns.distplot(dataTrain['SalePrice'])


# In[ ]:


# log transformation
dataTrain['SalePrice']=np.log1p(dataTrain['SalePrice'])
sns.distplot(dataTrain['SalePrice'])


# ***Ridge regression***
# 
# To apply ridge regression, I first need to transform categorial features so that they could be calculaed. Here, I use a function in the panda library to hot encode.
# 
# Then, I split the training set into two parts: one for regression training, one for testing result. Because ridge regression's accuracy depends on parameter alpha, I will test a range of alpha and then using this small testing set to test the result. In this way, I could choose the optimized model for ridge regression. This technique is called cross validation.

# In[ ]:


xTrainData = dataTrain.drop(['SalePrice'], axis=1)
inputData = xTrainData.append(dataTest)
# convert using hot encoding
inputData = pd.get_dummies(inputData)
xTrainLenth = xTrainData.shape[0]
xTrainData = inputData[0:xTrainLenth]
xTestData = inputData[xTrainLenth:]
print(xTrainData.shape, xTestData.shape)
# split the training set into set for model building and set for model optimizing
xTrain, xOpt, yTrain, yOpt = train_test_split(xTrainData, dataTrain['SalePrice'], test_size=0.3, random_state=100)


# In[ ]:


# iterate throw the parameter list for the best value of parameter alpha
listPara=[0.0001, 0.001, 0.01, 1, 10, 100, 1000]
# record the error for each parameter chosen
error = []
for i in range(len(listPara)):
    # apply ridge regression
    ridgeReg = Ridge(alpha=listPara[i], copy_X=True, fit_intercept=True)
    ridgeReg.fit(xTrain, yTrain)
    optPredict = ridgeReg.predict(xOpt)
    # calculate the error
    error.append(np.sqrt(mean_squared_error(optPredict, yOpt)))
# plot the error
plt.scatter(x=listPara, y=error, color='b', marker='*')
# calculate the best parameter in list
print(pd.Series(data=error, index=listPara).idxmin())


# In[ ]:


# iterate throw the parameter list for the best value of parameter alpha
listPara = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# record the error for each parameter chosen
error = []
for i in range(len(listPara)):
    # apply ridge regression
    ridgeReg = Ridge(alpha=listPara[i], copy_X=True, fit_intercept=True)
    ridgeReg.fit(xTrain, yTrain)
    optPredict = ridgeReg.predict(xOpt)
    # calculate the error
    error.append(np.sqrt(mean_squared_error(optPredict, yOpt)))
# plot the error
plt.scatter(x=listPara, y=error, color='b', marker='*')
# calculate the best parameter in list
print(pd.Series(data=error, index=listPara).idxmin())


# In[ ]:


# iterate throw the parameter list for the best value of parameter alpha
listPara = np.arange(1, 21, 1)
# record the error for each parameter chosen
error = []
for i in range(len(listPara)):
    # apply ridge regression
    ridgeReg = Ridge(alpha=listPara[i], copy_X=True, fit_intercept=True)
    ridgeReg.fit(xTrain, yTrain)
    optPredict = ridgeReg.predict(xOpt)
    # calculate the error
    error.append(np.sqrt(mean_squared_error(optPredict, yOpt)))
# plot the error
plt.scatter(x=listPara, y=error, color='b', marker='*')
# calculate the best parameter in list
print(pd.Series(data=error, index=listPara).idxmin())


# In[ ]:


# iterate throw the parameter list for the best value of parameter alpha
listPara = np.arange(6, 8, 0.1)
# record the error for each parameter chosen
error = []
for i in range(len(listPara)):
    # apply ridge regression
    ridgeReg = Ridge(alpha=listPara[i], copy_X=True, fit_intercept=True)
    ridgeReg.fit(xTrain, yTrain)
    optPredict = ridgeReg.predict(xOpt)
    # calculate the error
    error.append(np.sqrt(mean_squared_error(optPredict, yOpt)))
# plot the error
plt.scatter(x=listPara, y=error, color='b', marker='*')
# calculate the best parameter in list
print(pd.Series(data=error, index=listPara).idxmin())


# ***Apply ridge regression and produce output***
# 
# In the previous processes, I have found that ridge regression works best when alpha is 6.7. Therefore, I perform ridge regression again, now using the complete training set provided.
# 
# Using the regression model, I predict sale prices for the test set and output the results.

# In[ ]:


# apply ridge regression
ridge = Ridge(alpha=6.7)
# use all training data available
ridge.fit(xTrainData, dataTrain['SalePrice'])
dataPredicted = ridge.predict(xTestData)
# perform the counter-transformation for log-transformation
dataPredicted = np.exp(list(dataPredicted))-1

idDf = pd.DataFrame(pd.read_csv('../input/test.csv')['Id'])
dataPreDf = pd.DataFrame(dataPredicted, columns=['SalePrice'])
output = pd.concat([idDf, dataPreDf], axis=1)
outputResult = pd.DataFrame(output)
outputResult.to_csv('submission.csv', index=False)


# **Score**
# 
# The score of the submission is 0.12956.
# 
# 
# **Thank you for reading!**
# 
# **I hope that it helps. Please feel free to post comments about anything that I need to clarify, where I did wrongly, and where I need to improve.**
