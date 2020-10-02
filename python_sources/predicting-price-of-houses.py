#!/usr/bin/env python
# coding: utf-8

# Kaggle is an excellent platform to work on examples and improve your Machine Learning Skills. Last month, I started reading about Machine Learning and jumped into the example in Kaggle to apply my knowledge. 
# 
# The below example is for the dataset in House Prices: Advanced Regression Techniques on Kaggle. You have Train and Test data which has been provided and you need to submit the predicted prices for the Test dataset to Kaggle. 
# 
# I have a thing for Python, I guess she was my first girlfriend in the world of Machine Learning and as you all know it is very difficult to forget the first love. Hence here I have used Python to write my code. 
# 
# Kaggle Score - 0.14389 Public Leaderboard - 1023 (at time of finalising this notebook)
# 
# Numerical features have only been selected and feature selection has been done primarily on correlation coffecient. 

# In[ ]:


#Import the necessary Python Packages. 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib as plt 
import sklearn

# Set ipython's max row display
pd.set_option('display.max_row', 10000)

#Setting to print all the values in array
np.set_printoptions(threshold=np.nan)

# Set iPython's max column width to 50
pd.set_option('display.max_columns', 500)


# In[ ]:


#Import Dataset downloaded from the Kaggle https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

traindata = pd.read_csv('../input/train.csv')
testdata = pd.read_csv('../input/test.csv')


# In[ ]:


#Let us try to understand more about the data.
traindata.info()


# In[ ]:


total = traindata.isnull().sum().sort_values(ascending=False)
percent = (traindata.isnull().sum()/traindata.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data


# So we see that there are so columns which have lot of missing data. I would get rid of any such columns which have too many nulls as they seem to be very in-significant for the sale price prediction. I would get rid of 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature' columns here. For the rest we will decide if we want to fill them up with something, if we select them as a feature for our prediction. 

# In[ ]:


#Get ridding of the columns with lot of missing data
traindata = traindata.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)
testdata = testdata.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)


# Next, we would like to understand the select only the numerical features for our analysis for now. Let us filter out all the numerical columns and create a new dataset. In future I will also try to take into account the columns which are non-numeric and might impact the prices. We would use the Pearson correlation coefficient between the Sale Price and the numerical features.

# In[ ]:


#First Deal with the numerical variables then move to the categorical string variables.
#Create Data set with numerical variables
num_trainData = traindata.select_dtypes(include = ['int64', 'float64'])
numcol = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', 'SalePrice']


# In[ ]:


#Find out correlation with numerical features
traindata_corr = num_trainData.corr()['SalePrice'][:-1]
golden_feature_list = traindata_corr[abs(traindata_corr) > 0].sort_values(ascending = False)
print("Below are {} correlated values with SalePrice:\n{}".format(len(golden_feature_list), golden_feature_list))


# I would first consider the following columns: 
# 
# OverallQual      0.790982
# GrLivArea        0.708624
# GarageCars       0.640409
# GarageArea       0.623431
# TotalBsmtSF      0.613581
# 1stFlrSF         0.605852
# FullBath         0.560664
# TotRmsAbvGrd     0.533723
# YearBuilt        0.522897
# YearRemodAdd     0.507101
# 
# Now OverallQual, GrLivArea seem to have a very strong correlation with SalePirce. I would definetly consider them. 
# However before taking call on rest of the columns I would see if features are also correated to each other. So lets create a heatmap and see the results. 

# In[ ]:


#Create heatmap for correlated numerical variables
get_ipython().run_line_magic('matplotlib', 'inline')
traindata_corrheatmap = num_trainData.corr()
cols = traindata_corrheatmap.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(num_trainData[cols].values.T)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


# So above heatmap clearly shows thet Garage Cars and Garage Area are co-related to each other. Hence does not make sense to take both into Feature list. I will go for Garage Area. Similarly I see that 1stFlrSF and TotalBsmtSF are closely related. Also GrLiv Area and TotRmsAbvGrd seem to quite strongly Correlated. I would go with following features:
# 'OverallQual','MSSubClass', 'KitchenAbvGr','OverallCond', 'GrLivArea', 'EnclosedPorch', 'GarageArea','TotalBsmtSF', 
# 
# Now why MSSubClass and OverallCond ? A negative correlation means that there is an inverse relationship between two variables - when one variable decreases, the other increases. Hence the features. We can select other as well, but for now let's see how these look.
# 
# Next let's try to understand more about are star of the show 'SalePrice'. We would like to understand the distribution of data.
# 
# SalePrice does not look very normal here. There is 'peakedness', positive skewness and the data does not follow the diagonal line. Let us also try solve the problem here and see how does the data look. I also would like to see how does the data look like for GrLivArea and TotalBsmtSF 

# In[ ]:


#Understand the distribution of the Sale Price
traindata['SalePrice'].describe()


# In[ ]:


traindata['SalePrice'].skew()


# In[ ]:


traindata['SalePrice'].kurtosis()


# The Sale Prices dont seem to be evenly distributed and is deviating from normal distribution. Also the peakedness is quite high and has a positive skewness.

# In[ ]:


sns.distplot(traindata['SalePrice'], color = 'b', bins = 100)


# In[ ]:


from scipy import stats
import matplotlib.pyplot as plt
res = stats.probplot(traindata['SalePrice'], plot=plt)


# In[ ]:


sns.distplot(np.log(traindata['SalePrice']), color = 'r', bins = 100)


# In[ ]:


res = stats.probplot(np.log(traindata['SalePrice']), plot=plt)


# So what did above was, first I checked the distribution of the Sale Price Data which was available in the data set provided by Kaggle. And I see that the data is not normally distributed. 
# 
# So I log transformed the sale prices and checked the distribution agains. The data now looks normally distributed and we would keep this in mind when we go for our prediction. The Same approach has been taken for GrLivingArea and TotalBsmtSF. We would log transform the data for both these features as well. 

# In[ ]:


#Understand the behaviour of data in GrLivArea
from scipy import stats
import matplotlib.pyplot as plt
res = stats.probplot(traindata['GrLivArea'], plot=plt)


# In[ ]:


sns.distplot(traindata['GrLivArea'], color = 'b', bins = 100)


# In[ ]:


sns.distplot(np.log(traindata['GrLivArea']), color = 'b', bins = 100)


# In[ ]:


from scipy import stats
res = stats.probplot(np.log(traindata['GrLivArea']), plot=plt)


# In[ ]:


#Understand the skewness of TotalBsmtSF
sns.distplot(traindata['TotalBsmtSF'], color = 'b', bins = 100)


# In[ ]:


res = stats.probplot(traindata['TotalBsmtSF'], plot=plt)


# Let's also try to visualize the features we selected and see what kind of relation do they share with Sale Price. From the graphs you would see that they are quite linear in nature. I could also see some outliers there and upon deep diging I found that there were some properties which were quite huge but the prices were very low. I would rather delete those rows. We can also see what other outliers exist to improve the model in future. But for now let's move ahead. 

# In[ ]:


traindata.plot.scatter(x = 'GrLivArea', y = 'SalePrice')

traindata.plot.scatter(x = 'GarageArea', y = 'SalePrice')

traindata.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice')

traindata.plot.scatter(x = '1stFlrSF', y = 'SalePrice')

sns.boxplot(x = 'OverallQual', y = 'SalePrice', data = traindata)

sns.boxplot(x = 'GarageCars', y = 'SalePrice', data = traindata)

sns.boxplot(x = 'FullBath', y = 'SalePrice', data = traindata)

sns.boxplot(x = 'TotRmsAbvGrd', y = 'SalePrice', data = traindata)

sns.boxplot(x = 'YearBuilt', y = 'SalePrice', data = traindata)

sns.boxplot(x = 'YearRemodAdd', y = 'SalePrice', data = traindata)


# I think we are good for now. Based on our Exploratory data analysis I believe we could finalise on some of the features which we see have quite high impact on the Sale Prices. Let's go ahead and delete the outliers now and also log transform the features GrLivArea, SalePrices and TotalBsmtSF. We would also fill the missing values in both train and test dataset with some defaults for now. 
# 
# I have used the XGBoost and Linear Regression here. The values predicted by both the algorithms were then averaged out for the final predictions. The output was then generated in the form of a csv file and posted on Kaggle Leaderboard. Honestly, I think there is still lot of room to improve the algorithm. As I learn more techniques, I will come back and improve this piece. I intend to add some scoring sections in the code below to see the R SQuare, Adjusted R Square, OLS, Cross Validation Score. Hopefully I will do this very soon. 

# In[ ]:


#Delete the outliers
traindata = traindata.drop(traindata[traindata['Id'] == 1299].index)
traindata = traindata.drop(traindata[traindata['Id'] == 524].index)


# In[ ]:


#On basis of EDA we did earlier, filter out the variable we want to use for predicting the sale price
finaldata = traindata.filter(['OverallQual','MSSubClass', 'KitchenAbvGr','OverallCond', 'GrLivArea', 'EnclosedPorch', 'GarageArea','TotalBsmtSF',  'YearBuilt', 'SalePrice'], axis = 1)
finaltest = testdata.filter(['OverallQual','MSSubClass', 'KitchenAbvGr', 'OverallCond','GrLivArea', 'EnclosedPorch', 'GarageArea','TotalBsmtSF',  'YearBuilt'], axis = 1)


# In[ ]:


#Handle mising values in test data 
finaltest.loc[finaltest.GarageArea.isnull(), 'GarageArea'] = 0
finaltest.loc[finaltest.TotalBsmtSF.isnull(), 'TotalBsmtSF'] = 0


# In[ ]:


#Transform Sale Price and GrLivArea to reduce standardize the data 
finaldata['SalePrice'] = np.log(finaldata['SalePrice'])
finaldata['GrLivArea'] = np.log(finaldata['GrLivArea'])
finaltest['GrLivArea'] = np.log(finaltest['GrLivArea'])


# In[ ]:


#Find out the columns which are missing in final data 
total = finaldata.isnull().sum().sort_values(ascending=False)
percent = (finaldata.isnull().sum()/finaldata.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# In[ ]:


#Splt into predictor and variable
xtrain = finaldata.iloc[:, :-1].values
ytrain = finaldata.iloc[:,9].values
xtest = finaltest.iloc[:, :9].values


# In[ ]:


#Prediction Model
import xgboost as xgb
regr = xgb.XGBRegressor()
regr.fit(xtrain, ytrain)

#Calculate the score for the XGBoost Model
regr.score(xtrain,ytrain)

# Run predictions using XGBoost
y_pred = regr.predict(xtrain)

#Predict the prices for Test Data Set
y_test = regr.predict(xtest)


# In[ ]:


##Fit Linear Regression Model 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#Calculate score for the Linear Regression model
regressor.score(xtrain,ytrain)

#Predict Value of the house using Linear Regression
ytrainpred = regressor.predict(xtrain)

#Predict Value of the house on test data set 
ytestpred = regressor.predict(xtest)


# In[ ]:


#Average out the predicted value from XGBoost and Linear Regression
finalpred = (y_test+ytestpred)/2
finalpred = np.exp(finalpred)


# In[ ]:


#Output to csv

my_submission = pd.DataFrame(finalpred, index=testdata["Id"], columns=["SalePrice"])
my_submission.to_csv('submission.csv', header=True, index_label='Id')

