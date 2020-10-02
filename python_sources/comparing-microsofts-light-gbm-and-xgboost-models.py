#!/usr/bin/env python
# coding: utf-8

# Getting started: 
# * What Will i do in this notebook - 
# 
# First things First: **snacks and chip checked, Coffee checked**
# 
# Plan:
# * Introduction
# * Loading the data
# * Feature engineering
# * Finding correlation with predictor
#     * Finding the most correlated features
# * Imputting missing values
# * One hot encoding or pd.Dummies
# * Scaling 
# * Checking skewness

# #  ** Introduction**

# This notebook is a very simple and basic introductory to some concepts in machine learning and also introduce a relatively new algorithm that hasn't got a lot of reading resources on the internet except for its documentation. 
# 
# In a nut shell, Xgboost( eXtreme Gradient BOOSTing) is the predominant model here on Kaggle and it is also responsible for most of the top scores in many competitions due to its model perfromance. It is an implementation of the gradient boosting decision trees and is really fast when compared to other implementations of  gradient boosting. 
# 
# On the other hand is the Light GBM which was release by Microsoft in January 2017. It is  prefix ''Light'' because of its high speed. It is getting more popular because as the size of data get bigger everyday, it is becoming difficult for more traditional data science algorithms to handle these large dataset, produce faster results, use less memory and also give us good prediction accuracy.  
# 
# I am also quite a newcomer to the Kaggle scene and this is my first implementation of Light GBM. Therefore, i am pretty sure there is room for improvement so please feel freee to leave any comment or suggestions on how it can be improved. \
# 
# The materials in this notebook( part 1 especially) borrows heavily form some awesome notebooks i stumble upon. I use traditional models in this part. Part 2 is the comparative study of the Light GBM and xgboost.

# In[ ]:


#importing the necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
#import lightgbm as lgb
#import xgboost as xgb


# # 1 - LOADING THE DATA

# In[ ]:


#Read the necessay train and test .csv files
train = pd.read_csv("../input/train.csv")
print('The size of the train dataset is {}'.format(train.shape))

test = pd.read_csv("../input/test.csv")
print('The size of the test dataset is {}'.format(test.shape))


# In[ ]:


#display th first 5 rows of the train and test sets
train.head()


# In[ ]:


test.head()


# We see that in both the training and the test sets, we have an 'Id- column' which we do not need right now. So we are going to save these columns and use them later in the submission files. Then we drop the id columns from boht datasets.

# In[ ]:


#extracting the id columns form train and test datasets
id_train = train['Id']
id_test = test['Id']


# In[ ]:


#removing them form the train and test sets
train.drop('Id', axis = 1, inplace= True)
test.drop('Id', axis = 1, inplace = True)
#we check to see that they are gone


# In[ ]:


test.head()


# In[ ]:


test.shape


# In[ ]:


# We take a look at the states
train.describe()


# Obsevations:  #Let me know of other observations which i missed :)
# 
#  We see that the
#  * The counts for features are not the same meaning, and we know that the total count for the train data should be 1460 rows. This means some of the features have missing values or NaNs.
#  * The difference betwenn the min and max values is way too large for some features, meaning these features have lots of outliers.  When outliers are left in a dataset, the models may because of hyperensitivity to these points resulting in an over or under fit model. But we have to be smart about the way we handle outlier. It is not always safe to remove them!!!

# # 2 - Analysis and Feature engineering

# In[ ]:


#we take a look at the different data types present in the train data
train.dtypes.value_counts()


# This means we have 43 categorical features (object = 43) and the rest are numerical features for the train and test datasets. 

# In[ ]:


test.dtypes.value_counts()


# In[ ]:





# Th facilitate the feature engineering, lets merge the train and test data into 1 dataFrame which we will call 'DATA_ALL' but we exclude the Saleprice columns since we do not want to alter our Predictor 
# 

# In[ ]:


#we keep this for when we will be separating DATA_ALL back into train and test 

train_rowsize = train.shape[0]
test_rowsize = test.shape[0]
test_rowsize   


# In[ ]:


train_rowsize


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

data_all = pd.concat((train, test))
data_all.drop('SalePrice', axis = 1, inplace = True)


# In[ ]:


data_all.head()


# In[ ]:


#we check the size of the new dataframe 
print('The shape of the data_all is:  {} '.format(data_all.shape))


# In[ ]:


#Here is a list of all the features with Nans and the number of null for each features
null_values = data_all.columns[data_all.isnull().any()]
null_features = data_all[null_values].isnull().sum().sort_values(ascending = False)
missing_data = pd.DataFrame({'No of Nulls' :null_features})
missing_data


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('dark')

plt.figure(figsize= (16, 8))
plt.xticks(rotation='90')
ax = plt.axes()
sns.barplot(null_features.index, null_features)
ax.set(xlabel = 'Features', ylabel = 'Number of missing values', title = 'Missing data');


# In[ ]:


# Correlation between the features and the predictor- SalePrice
predictor = train['SalePrice']
fields = [x for x in train.columns if x != 'SalePrice']
correlations = train[fields].corrwith(predictor)
correlations = correlations.sort_values(ascending = False)
# correlations
corrs = (correlations
            .to_frame()
            .reset_index()
            .rename(columns={'level_0':'feature1',
                                0:'Correlations'}))
corrs


# In[ ]:


plt.figure(figsize= (16, 8))
ax = correlations.plot(kind = 'bar')
ax.set(ylabel = 'Pearson Correlation', ylim = [-0.2, 1.00]);


# In[ ]:


# Get the absolute values for sorting
corrs['Abs_correlation'] = corrs.Correlations.abs()
corrs


# A Histogram of absolute correlations

# In[ ]:


plt.figure(figsize= (16, 8))
sns.set_context('talk')
sns.set_style('white')
sns.set_palette('dark')

ax = corrs.Abs_correlation.hist(bins= 35)

ax.set(xlabel='Absolute Correlation', ylabel='Frequency');


# In[ ]:


# Most correlated features wrt the abs_correlations
corrs.sort_values('Correlations', ascending = False).query('Abs_correlation>0.45')


# # Imputing the values.  https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard does an excellent job- check it out!

# Having  missing values or NaNs in your dataset can couse error with come machine learning algorithms. We could try to replace the missing values with 
# * A constant value that has meaning within the domain like a 0, or 
# * We could replace them with the mean, median or mode of values 
# * With another values selected from a random record or another predicted model
# 
# We will start imputing the missing values in our dataset(DATA):

# For some of these features, we are going to replace the mssing values by 0. For example: 
# * **PoolQc** data description says NA means "No Pool". Which is common for homes to not have pools so we replace with 0. Same for **Alley, Fences, fireplaceQu, Misc, etc...**
# 

# In[ ]:


missing_data = ['PoolQC',"MiscFeature","Alley", "Fence", "FireplaceQu", 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                  'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 
                  'BsmtHalfBath', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', "MasVnrType", "MasVnrArea",
                  'MSSubClass']

for x in missing_data:
    data_all[x] = data_all[x].fillna(0)


# In[ ]:


# null_values_2 = data_all.columns[data_all.isnull().any()]
# null_features_2 = data_all[null_values_2].isnull().sum().sort_values(ascending = False)
# missing_data_2 = pd.DataFrame({'No of Nulls' :null_features_2})
# missing_data_2


# #Now we deal with the missing data in the lotFrontage
# * ** LotFrontage** : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
# 
# 

# In[ ]:


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
data_all["LotFrontage"] = data_all.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# For the remaining features with missing values, we will replace the missing values with the most frequent values in that column.

# In[ ]:


data_all['MSZoning'].value_counts(normalize = True)  
#we see that in the MsZoning 77% of the data is RL. so we replace the Missing values here with RL


# In[ ]:


data_all['MSZoning'] = data_all['MSZoning'].fillna('RL')
data_all['MSZoning'].isnull().any()    # to see if the MSZoning has any missing values(NaNs)


# In[ ]:


# Utilities
data_all['Utilities'].value_counts(normalize = True)


# In[ ]:


data_all['Utilities'] = data_all['Utilities'].fillna('AllPub')


# In[ ]:


# Functional
data_all['Functional'].value_counts(normalize = True)


# In[ ]:


data_all['Functional'] = data_all['Functional'].fillna('Typ')


# In[ ]:


data_all['Electrical'] = data_all['Electrical'].fillna(data_all['Electrical'].mode()[0])
data_all['KitchenQual'] = data_all['KitchenQual'].fillna(data_all['KitchenQual'].mode()[0])
data_all['Exterior1st'] = data_all['Exterior1st'].fillna(data_all['Exterior1st'].mode()[0])
data_all['Exterior2nd'] = data_all['Exterior2nd'].fillna(data_all['Exterior2nd'].mode()[0])
data_all['SaleType'] = data_all['SaleType'].fillna(data_all['SaleType'].mode()[0])


# In[ ]:


null_values_2 = data_all.columns[data_all.isnull().any()]
null_features_2 = data_all[null_values_2].isnull().sum().sort_values(ascending = False)
missing_data_2 = pd.DataFrame({'No of Nulls' :null_features_2})
missing_data_2

print('|\t\t NO MORE MISSING VALUES REMAINING. \n\n\t\t\t...IMPUTING COMPLETED ...')


# # One Hot Encoding and Scaling

# In[ ]:


data_all.dtypes.value_counts()


# Now that we have done the imputing on all the data. We now separate the data into their original train and test columns

# In[ ]:


train_new = data_all[:train_rowsize]
test_new = data_all[train_rowsize:]
test_new.shape


# In[ ]:


train_new.dtypes.value_counts()


# In[ ]:


test_new.dtypes.value_counts()


# In[ ]:


train_new.head()


# As we can see from above the  new train and test sets have 43 categorical data and the rest of the data is numerical. We are going to encode the categorical features alone. So we need to separate the categoricals from the numerocal features.

# Separatiig the features into numerical and catergorical data will help encode the categorical data easily. We are going to isolate all the object/categorical feature and converting them to numeric features

# In[ ]:


#This is the separation of features into numerical and catergorical features, to do
#feature engineering on each class of data.

#isolating all the object/categorical feature and converting them to numeric features

train_numericals = train[train_new.select_dtypes(exclude = ['object']).columns]
test_numericals = test[test_new.select_dtypes(exclude = ['object']).columns]

#takeoutthe salesprice from the numerical features
#train_numericals = train_numericals.drop("SalePrice")
train_categcols = train_new.select_dtypes(include = ['object']).columns
test_categcols = test_new.select_dtypes(include = ['object']).columns

train_categoricals = train[train_categcols]
test_categoricals = test[test_categcols]

# train_numeric = train[numerical_cols]
# test_numeric = test[numerical_cols2]

print("Shape of Train Categoricals features : {}".format(train_categoricals.shape))
print("Shape of Train Numerical features : {}\n".format(train_numericals.shape) )

print("Shape of Test Categoricals features : {}".format(test_categoricals.shape))
print("Shape of Test Numerical features : {}".format(test_numericals.shape) )


# We will use the pd.Dummies method to encode the features. why? I prefer it ;)

# In[ ]:


# Do the one hot encoding on the categorical features
train_dummies = pd.get_dummies(train_new, columns = train_categcols)
test_dummies = pd.get_dummies(test_new, columns = test_categcols)
#align your test and train data
train_encoded, test_encoded = train_dummies.align(test_dummies, join = 'left', axis = 1)
print('\t\tShape of the new encoded train: {}'.format(train_encoded.shape))
print('\n\t\tShape of the new encoded test: {}'.format(test_encoded.shape))
print('\n\t\t\t....Encoding completed.....')


# In[ ]:


train_encoded.dtypes.value_counts()


# # SKEWNESS

# Lets take alook at how skewed our dataset is. Just a little background:
# 
# * In statistics, skewness is a measure of the asymmetry of the probability distribution of a random variable about its mean. In other words, skewness tells you the amount and direction of skew (departure from horizontal symmetry). The skewness value can be positive or negative, or even undefined. If skewness is 0, the data are perfectly symmetrical, although it is quite unlikely for real-world data. As a general rule of thumb:
# 
# * If skewness is less than -1 or greater than 1, the distribution is highly skewed.
# * If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.
# * If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.
# 
# references:https://help.gooddata.com/display/doc/Normality+Testing+-+Skewness+and+Kurtosis

# In[ ]:


#we check for skewness in the float data

skew_limit = 0.75
skew_vals = train_numericals.skew()

skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skewness'})
            .query('abs(Skewness) > {0}'.format(skew_limit)))

skew_cols 


# I will be using a numpy  log1p to transform  the very skewed data. Just to give a sense of what numpy log1p tranformation does in a skewed dataset, We are going to design a before and after log1p .

# In[ ]:


tester = 'LotArea'
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(16,5))
#before normalisation
train_new[tester].hist(ax = ax_before)
ax_before.set(title = 'Before nplog1p', ylabel = 'Frequency', xlabel = 'Value')

#After normalisation
train_new[tester].apply(np.log1p).hist(ax = ax_after)
ax_after.set(title = 'After nplog1p', ylabel = 'Frequency', xlabel = 'Value')

fig.suptitle('Field "{}"'.format(tester));


# The before normalisation shows a right skewed distribution or positive skewed feature. After nplog1p, the distribution is more symmetrical.
# 
# **BTW: You can change the tester and see the different features before and after nplog1p**
# 

# In[ ]:


print(skew_cols.index.tolist()) #returns a list of the values


# In[ ]:


#Log transfrom all the numerical features except the Salepice column
for col in skew_cols.index.tolist():
    train_encoded[col] = np.log1p(train_encoded[col])
    test_encoded[col]  = test_encoded[col].apply(np.log1p)  # same thing
print(test_encoded.dtypes.value_counts())
print ('\n\t\t:) Skewed data Transformation Completed :)')


# Lets take a look at the predictor('SalePrice')

# In[ ]:


#plotting the distribution curve for the SalePrice
f, ax = plt.subplots(figsize=(12, 6))
#plt.xticks(rotation='90')
sns.distplot(train['SalePrice']);


# The Saleprice is positively skewed- it is tilting more towards the right. It is usally recommended to fix the skewness of such distribution to get a better model. Some machine learning algorithms work better with features that are normally distributed i.e symmetrical and bell-shaped like the one below. Also,  It is usually adviced to use the same transformation used in the train data on the predictor

# In[ ]:


predictor = np.log1p(train.SalePrice)
#plotting the distribution curve for the SalePrice
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(predictor);


# Do you see the difference?. Now the distribution in more symmetric, and more towards the center.

# # Predictive modelling

# In[ ]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(train_encoded, predictor, 
                                                    test_size=0.3, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


#We creat a function for calculating the mean_squared_erros
from sklearn.metrics import mean_squared_error

def rmse (true_data, predicted_data):
    return np.sqrt(mean_squared_error(true_data, predicted_data))


# In the firstt part of modelling, i will be using the Linear regression, ridgecv, lassocv, elasticnetcv and  random forest. In the part 2, i really want to focus on applying XGBoost and Gradient Boosting Algorithms on tis data and see how both models differ from each other. 

# # Modelling (part 1)

# In[ ]:


#now to the most fun part. Feature engineering is over!!!
#i am going to use linear regression, L1 regularization, L2 regularization and ElasticNet(blend of L1 and L2)

#LinearRegression
linearRegression = LinearRegression().fit(X_train, y_train)
prediction1 = linearRegression.predict(X_test)
LR_score = linearRegression.score(X_test, y_test)
LR_rmse = rmse(y_test, prediction1)
print('The scoring and root mean squared error for Linear Regression in percentage\n')
print('\t\tThe score is: ',LR_score*100)
print('\t\tThe rmse is : ',LR_rmse*100)


# The RidegeCV, LassoCV, ElasticnetCV are Regression models with built-in cross validation so by default, it performs generalised cross-validation meaning it is going to go through all the alphas and choose the one value that perfromed best.

# In[ ]:


#choose some values of alpha for cross validation.
alphas = [0.005, 0.05, 0.1, 1, 5, 10, 50, 100]


# **RidgeCV uses a L2 regularization to reduce the magnitudes of the coefficients whichcan be helpful in situations where there is high variance **

# In[ ]:



#ridge
ridgeCV = RidgeCV(alphas=alphas).fit(X_train, y_train)
prediction2 = ridgeCV.predict(X_test)
R_score = ridgeCV.score(X_test, y_test)
R_rmse = rmse(y_test, prediction2)
print('The scoring and root mean squared error for Linear Regression in percentage\n')
print('\tThe parameter used for here was alpha = {}\n'.format(ridgeCV.alpha_))
print('\t\tThe score is: ',R_score*100)
print('\t\tThe rmse is : ',R_rmse*100)


# **LassoCV uses a L1 regularization to reduce the magnitudes of the coefficients. L1 regularization will selectively shrink some coefficients, effectively performing feature elimination. LASSO IS VERY SLOW SO I WILL USE JUST A FEW ALPHAS**

# In[ ]:


#lasso
lassoCV = LassoCV(alphas=[0.005, 0.001, 0.05, 0.01,1, 5], max_iter=1e2).fit(X_train, y_train)
prediction3 = lassoCV.predict(X_test)
L_score = lassoCV.score(X_test, y_test)
L_rmse = rmse(y_test, prediction3)
print('The scoring and root mean squared error for Linear Regression in percentage\n')
print('\tThe parameter used for here was alpha = {}'.format(lassoCV.alpha_))
print('\n\t\tThe score is: ',L_score*100)
print('\t\tThe rmse is : ',L_rmse*100)


# ElasticnetCV is the combinaision of L1 and L2 rgularisation. We need to set another parameter called the l1_ratio. 
# 
# ** Note** that a good choice of list of values for l1_ratio is often to put more values close to 1 (i.e. Lasso) and less close to 0 (i.e. Ridge) 
# 
# check [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html](http://) for more info

# In[ ]:


#elasticNetCV
l1_ratios = np.linspace(0.1, 0.9, 9)
elasticnetCV = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, max_iter=1e2).fit(X_train, y_train)
prediction4 = elasticnetCV.predict(X_test)
EN_score = elasticnetCV.score(X_test, y_test)
EN_rmse = rmse(y_test, prediction4)
print('The scoring and root mean squared error for Linear Regression in percentage\n')
print('\tThe parameter used for here was alpha = {} and l1_ratios = {} \n'.format(elasticnetCV.alpha_, elasticnetCV.l1_ratio_))
print('\t\tThe score is: ',EN_score*100)
print('\t\tThe rmse is : ',EN_rmse*100)


# I will use GridSearchCV here just to be consistent and also because random forest does not have a built in cross validation.

# In[ ]:


randfr = RandomForestRegressor(random_state = 42) #random_state to avoid the result from fluctuating


# In[ ]:


param_grid = { 
    'n_estimators': [50,250,500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [2, 4, 6, 8, 10],
}


# In[ ]:


randfr_cv = GridSearchCV(estimator=randfr, param_grid=param_grid, cv= 5)  #cv = 5 to specify the number of folds(5 in this case)  in a stratified Kfold
randfr = randfr_cv.fit(X_train, y_train)


# In[ ]:


prediction5 = randfr.predict(X_test)
#print(prediction5.shape)
RF_score = randfr.score(X_test, y_test)
RF_rmse = rmse(y_test, prediction5)
print('The scoring and root mean squared error for Linear Regression in percentage\n')
print('\tThe parameter used for here were = {}\n'.format(randfr_cv.best_params_))
print('\t\tThe score for random forest is: {} '.format(RF_score*100))
print('\t\tThe rmse is: {} '.format (RF_rmse*100))


# In[ ]:


#putting it lall together

score_vals = [LR_score, R_score, L_score, EN_score, RF_score]
rmse_vals = [LR_rmse, R_rmse, L_rmse, EN_rmse, RF_rmse]
labels = ['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'RandomForest']

rmse_df = pd.Series(score_vals, index=labels).to_frame()
rmse_df.rename(columns={0: 'SCORES'}, inplace=1)
rmse_df['RMSE'] = rmse_vals
rmse_df


# In[ ]:


rmse_df = rmse_df.sort_values(['RMSE'], ascending=True)
rmse_df


# # Modelling part 2

# Here i want to do a comparative study of XGboost And the Light GBM. 
# 
# As you may already know, XGBoost is one of the top models uses in competitions and it is the model that usually produces the best and top score here on Kaggle. This is because it is fast effective and very powerful. 
# 
# What about LightGBM?
# 
# Sincerely, i had never heard of it before until i stumble upon it here [https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard](http://). So i decided to dig a little deeper, and try to understand their differences with respect to this dataset. 
# 
# 

# These blogs explain the 2 concept very well. Take a look to learn more:
# * https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db
# * https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
# 
#     Another useful paper about this by Microsoft research: **You definately want to read this**
# * https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf
# 
# **Below is a little summary i could gather from reading the paper:**

# # * Light GBM Versus XGBoost

# *** XGBoost**:
# 
# XGBoost is one of the most popular algorithms cout there and it has even become the de-facto algorithm for winning competitions here on Kaggle. But its efficientcy and scalability is still insatisfactory especially  when dealing with larger datasets and large feature dimensionality. The reason being that in **XGBoost, for each feature, all the data instances(observations/samples) need to be scanned  to estimate the information gain of all possible split points, and this can be very time consuming especially when handling big data as 
# computational complexity is propartional to the number of features and the number of samples.**
# 
# 
# *** How Does XGBoost work:**
# 
# It works by using both the  pre-sorting algorithm and the histogram based algorithms for finding the best split. the pre sorting algorithm is one of the most popular algorithms for finding split points but it is inefficient in both training speed and memory consumption. Pre-sorting algorthm works by 
# 
# 1.  It enumerates all the features for each node  
# 2. Sort the instance or samples for eeach features by feature values
# 3. Use a linear scan to decide the best split along that feature basis 
# 4. Finally, it takes the best split solution along all the features
# 
# 
# *** Light GBM:**
# 
# Microsoft released it first stable version of Light GBM in Jan 2017. To remedie this problem, Light GBM uses a technique called GOSS( Gradient-based One-Side Sampling) which besically achieves a good balance between reducing the number of instances while maintainig the accuracy for learned decision trees.
# 
# *** How does Light GBM work**
# 
# While there is no native weight for data instance in Gradient Boosting Decision Trees, data instances with different gradients play different roles in the computation of information gain. In particular, according to the definition of information gain, those instances with larger gradients (i.e., under-trained instances) will contribute more to the information gain. Therefore, when down sampling the data instances, in order to retain the accuracy of information gain estimation, we should better keep those instances with large gradients (e.g., larger than a pre-defined threshold, or among the top percentiles), and only randomly drop those instances with small gradients
# 
# **In short: GOSS keeps all the instances with large gradients and performs random sampling on the instances with small gradients.**

# In[ ]:


from datetime import datetime

start_xgb = datetime.now()

xgb = XGBRegressor().fit(X_train, y_train)

end_xgb = datetime.now()

xgb_time = end_xgb - start_xgb
print('Duration for XGBoost: {}'.format(xgb_time))


# In[ ]:


prediction6 = xgb.predict(X_test)
xgb_score = xgb.score(X_test, y_test)
xgb_rmse = rmse(y_test, prediction6)
print('The scoring and root mean squared error for XGBoost in percentage\n')
print('\t\tThe score is: ',xgb_score*100)
print('\t\tThe rmse is : ',xgb_rmse*100)


# In[ ]:





# In[ ]:


Adding_xgboost = pd.Series({'SCORES': xgb_score, 'RMSE': xgb_rmse}, name = 'XGBoost')
rmse_df = rmse_df.append(Adding_xgboost)
rmse_df


# In[ ]:


start_lgbm = datetime.now()

lgb = LGBMRegressor().fit(X_train, y_train)

end_lgbm = datetime.now()

lgbm_time = end_lgbm - start_lgbm
print('Duration for Light GBM: {}'.format(lgbm_time))


# In[ ]:


prediction7 = lgb.predict(X_test)
lgb_score = lgb.score(X_test, y_test)
lgb_rmse = rmse(y_test, prediction7)
print('The scoring and root mean squared error for light GBM in percentage\n')
print('\t\tThe score is: ',lgb_score*100)
print('\t\tThe rmse is : ',lgb_rmse*100)


# In[ ]:


Adding_lgbm = pd.Series({'SCORES': lgb_score, 'RMSE': lgb_rmse}, name = 'Light GBM')
rmse_df.append(Adding_lgbm)


# In[ ]:


print('\t\tComparing the 2 model durations:\n')
print('XGBOOST : {} \t\t LIGHT GBM : {}'.format(xgb_time, lgbm_time))


# In[ ]:


comparisons = {'Scores': (lgb_score, xgb_score), 'RMSE': (lgb_rmse, xgb_rmse), 'Execution Time' : (lgbm_time, xgb_time)}
comparisons_df = pd.DataFrame(comparisons)


# In[ ]:


comparisons_df.index= ['LightGBM','XGBOOST'] 
comparisons_df


# We can observe that, LightGBM significantly outperforms XGBoost in terms of speed but XGBoost has a slightly better score than LightGBM for this experiement. 
# 
# **So which do you prefer? Speed  or efficiency? i would like to hear from you so please let me know in the comment section and please add a reason for your choice. **
# 
#  **:-) Thanks for reading till the end. If you found this kernel helpful, an upvote would be highly appreciated. :-)**

# In[ ]:


test_encoded.isnull().sum()


# In[ ]:


null_values = test_encoded.columns[test_encoded.isnull().any()]
null_features = test_encoded[null_values].isnull().sum().sort_values(ascending = False)
missing = pd.DataFrame({'No of Nulls' :null_features})
missing


# In[ ]:


test_encoded = test_encoded.fillna(0)


# In[ ]:


prediction = lassoCV.predict(test_encoded) # WE USE THE BEST RMSE which is tht for Lasso
final_prediction = np.exp(prediction) #undoing the np log we did on the saleprices else the resu


# In[ ]:



House_submission = pd.DataFrame({'Id': id_test, 'SalePrice': final_prediction})
print(House_submission.shape)
House_submission.to_csv('House_prediction.csv', index = False)


# In[ ]:


print(House_submission.sample(6))


# # References/ useful kernels
# * https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# * https://www.kaggle.com/agodwinp/stacking-house-prices-walkthrough-to-top-5
# * https://www.kaggle.com/chocozzz/beginner-challenge-house-prices?scriptVersionId=7418555
# * https://www.kaggle.com/sociopath00/random-forest-using-gridsearchcv
# * https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db

# In[ ]:




