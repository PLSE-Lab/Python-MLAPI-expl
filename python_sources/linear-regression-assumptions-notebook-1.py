#!/usr/bin/env python
# coding: utf-8

# # Let's start journey in to Machine Learning with Linear Regression

# 
# **This is very important data set for beginners who are starting journey in to Machine Learning field.**
# 
# * We will start from  very first regression algorithm 'Linear Regression' from scratch.
# * Before applying Linear Regression we need to check for it's assumptions !!
# * We will check if all the Linear Regression assumptions are satisfied on the given data set.
# * If not satisfied then how to apply different techniques to make all the Linear regression assumption satisfied.
# 
# **Steps we are going to Follow:**
# 
# 1. Data Reading : Collect the data from input data set
# 2. Explore the data
# 3. Assumption of Linear Regression
# 4. Apply different techniques to make data satisfy assumptions.
# 

# # Assumptions of Linear Regression
# 
# 1. Linear relationship between each independent variable and the dependent variable
# 2. No or little Multicollinearity - No or Little Linearity between Predictors 
# 3. Homoscedasticity ( Constant Error Variance )
# 4. Independence of Errors ( vs Autocorrelation )
# 5. Multivariate Normality ( Normality of Errors )

# In[ ]:


# Importing all library to use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# # **Let's start the fun**

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input/house-prices-advanced-regression-techniques"]).decode("utf8")) #check the files available in the directory


# # Load the data

# In[ ]:


# Reading data from file to proes the s
# file is at same location where we have this python code sheet
# Using panda lib to read file
trainData = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
trainData.shape


# testData = pd.read_csv("/Users/ajaychauhan/Rajnish-GIThub-Repo/ML_Share_Doc-master/Code/LinearRegression/LR_Practice/HousePricingData/test.csv")
# testData.shape

# In[ ]:


# Informtion about dataset 
# Number of colums
# how many rows dataset have
# how mnay not-null value each column have
trainData.info()


# In[ ]:


# first five rows to view
# change value from 5 to any number to view more data
trainData.head(5)


# In[ ]:


# Provide statistics information about quantitative data
# 
trainData.describe()


# # Assumptions of Linear Regression
# 
# 1. Linear relationship between each independent variable and the dependent variable
# 2. No or little Multicollinearity - No or Little Linearity between Predictors 
# 3. Homoscedasticity ( Constant Error Variance )
# 4. Independence of Errors ( vs Autocorrelation )
# 5. Multivariate Normality ( Normality of Errors )
# 
# **Lets Check one by one if all the assumptions are satisfied**
# 

# # Let's check Linear Relationship First 
# ** High CoRelation between Target and Predictors**
# All the features which are highly co-related should be considered in Linear Regression and rest of all should be discarded

# In[ ]:


#correlation matrix for all Important Features
import seaborn as sns
corrmat = trainData.corr()
f, ax = plt.subplots(figsize=(15,12))
sns.heatmap(corrmat,cmap="Blues", square=True);


# In[ ]:


## Getting the correlation of all the features with target variable. 
(trainData.corr()**2)["SalePrice"].sort_values(ascending = False)[1:]


# TOP 10 important features
# 1. OverallQual      0.625652
# 1. GrLivArea        0.502149
# 1. GarageCars       0.410124
# 1. GarageArea       0.388667
# 1. TotalBsmtSF      0.376481
# 1. 1stFlrSF         0.367057
# 1. FullBath         0.314344
# 1. TotRmsAbvGrd     0.284860
# 1. YearBuilt        0.273422
# 1. YearRemodAdd     0.257151

# In[ ]:


# correlation matrix for 10 important features
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(trainData[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.1f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# # Lets Visulaize all important features one by one

# **SalePrice vs OverallQual**
# 
# OverallQual is a categorical variable, and a scatter plot is not the best way to visualize categorical variables. However, there is an apparent relationship between the two features. The price of the houses increases with the overall quality.
# 

# In[ ]:


def customized_scatterplot(y, x):
        ## Sizing the plot. 
    plt.style.use('fivethirtyeight')
    plt.subplots(figsize = (12,8))
    ## Plotting target variable with predictor variable(OverallQual)
    sns.scatterplot(y = y, x = x);


# In[ ]:


customized_scatterplot(trainData.SalePrice, trainData.OverallQual)


# * In Above graph we can see that as Overall Qual is increasing Price is also increasing.
# * Means we have linear relationship betwenn bothe variables

# # SalePrice vs GrLivArea
# * There we can see linear relationship
# * Also As you can see, there are two outliers in the plot below.

# In[ ]:



customized_scatterplot(trainData.SalePrice, trainData.GrLivArea)


# Here we can see that there are few outliers for GrLivArea apprx. values > 4500---> Lets check them out

# In[ ]:


trainData[trainData.GrLivArea>=4500]


# > SalePrice vs GarageArea
# * There we can see linear relationship
# * Also As you can see, there are four outliers in the plot below for garage area > 1200

# In[ ]:


customized_scatterplot(trainData.SalePrice, trainData.GarageArea);


# In[ ]:


trainData[trainData.GarageArea>=1200]


# **Scatter plots between 'SalePrice' and all highly correlated features**
# 
# * Below we can see that for all the highly corelated features there is high linear realtionship i.e 
# * One is increating other is decreasing :- Postive Corelation
# * Or one is increasing another is aslo increasing :- Negative Corealtion

# In[ ]:


#More visulization
#Lets check with scatter plot

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(trainData[cols], height = 2.5)
plt.show();


# # Residual plot for Heteroscedasticity
# * Let's look at the residual plot for independent variable GrLivArea and our target variable SalePrice .
# *  Ideally, if this assumptions are met, the residuals will be randomly scattered around the centerline of zero with no apparent pattern.
# *  The residual plot looks more like a funnel. The error plot shows that as GrLivArea value increases, the variance also increases, which is the characteristics known as Heteroscedasticity. 

# In[ ]:


plt.subplots(figsize = (12,8))
sns.residplot(trainData.GrLivArea, trainData.SalePrice);


# in the above scatter plot we can see that shape is funnel type that means theere is Heteroscedasticity in the data

# # Multivariate Normality ( Normality of Errors)
# * The linear regression analysis requires the dependent variable to be multivariate normally distributed. 
# * A histogram, box plot and Q-Q-Plot can check if the target variable is normally distributed.
# 
# 1. Histogram: Data should ne normalized
# 2. QQ diagram should be on 45 degree line for no Normal data
# 
# #**Lets Check Distribution of Sales_Price**

# In[ ]:


#histogram and normal probability plot
from scipy.stats import norm
from scipy import stats
sns.distplot(trainData['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(trainData['SalePrice'], plot=plt)


# **The SalePrice is skewed to the right. This is a problem because most ML models don't do well with non-normally distributed data. We can apply a log(1+x) tranform to fix the skew**

# In[ ]:


#SalePrice' is not normal. It shows 'peakedness', positive skewness and does not follow the diagonal line
# Lets apply log transformation

trainData['SalePrice'] = np.log(trainData['SalePrice'])


# In[ ]:


sns.distplot(trainData['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(trainData['SalePrice'], plot=plt)


# 1. Now we can see Sale Value column data is now normalzied.
# 1. This normalization we can apply to every column

# In[ ]:


trainData['GrLivArea'] = np.log(trainData['GrLivArea'])


# In[ ]:


plt.subplots(figsize = (12,8))
sns.residplot(trainData.GrLivArea, trainData.SalePrice);


# # No or Little multicollinearity:
# Multicollinearity is when there is a strong correlation between independent variables. Linear regression or multilinear regression requires independent variables to have little or no similar features.
# Multicollinearity can lead to a variety of problems

# In[ ]:


# Lets check CoRelation

## Plot fig sizing. 
import matplotlib.style as style
import seaborn as sns

style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize = (30,20))
## Plotting heatmap. 

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(trainData.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


sns.heatmap(trainData.corr(), 
            cmap=sns.diverging_palette(20, 220, n=200), 
            mask = mask, 
            annot=True, 
            center = 0, 
           );
## Give title. 
plt.title("Heatmap of all the Features", fontsize = 30);


# * There is 0.83 or 83% correlation between GarageYrBlt and YearBuilt.
# * 83% correlation between TotRmsAbvGrd and GrLivArea.
# * 89% correlation between GarageCars and GarageArea.
# * Similarly many other features such asBsmtUnfSF, FullBath have good correlation with other independent feature.
# * 
# * It also becomes clear the multicollinearity is an issue. 
# * For example: the correlation between GarageCars and GarageArea is very high (0.89), and both have similar (high) correlations with SalePrice. The other 6 six variables with a correlation higher than 0.5 with SalePrice are: -TotalBsmtSF: Total square feet of basement area 
# * 1stFlrSF: First Floor square feet 
# * FullBath: Full bathrooms above grade 
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms) 
# * YearBuilt: Original construction date 
# * YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)

# **You have remove all the coumns which as high collinearlity one by one manually**

# When you have checked all the assumptions and make your data set satisfying the linear assumptions.
# You are ready to go ahead and apply your Linear Regression.
# 
# **I will be uploading another notebook soon for:**
# 
# * Feature Enginnering
# * Data preprecessing
# * Handling Missing Values
# * Scaling of Data
# * Convert Categorical to Numerical data
# * Model Building
# * Overfitting/Underfitting ( Bias &. variance)
# * Cross Validation
# * Regularization (Ridge/lasso - L1/L2)
# * Prediction and Submission of Result
# 
# STAY TUNED......
# 
# 
# 

# # So This was very easy. is't it
# 
# I will be uploading another notebooks soon for:
# 
# * Feature Enginnering
# * Data preprecessing
# * Handling Missing Values
# * Scaling of Data
# * Convert Categorical to Numerical data
# * Model Building
# * Overfitting/Underfitting ( Bias &. variance)
# * Cross Validation
# * Regularization (Ridge/lasso - L1/L2)
# * Prediction and Submission of Result
# * STAY TUNED.....
# 
# 
# if you like the notebook and it helps you to start your jounery in to machine learning.
# Like and share..
# 
# Keep Learning and Keep Kaggling
