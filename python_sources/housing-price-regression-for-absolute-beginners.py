#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


import numpy as np 
import pandas as pd 

import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")

from scipy import stats
from scipy.stats import norm, skew #for some statistics

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split #splitting the data for train and test 

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# ## Load datasets and Inspect it

# In[ ]:


#loading the training and testing data sets 
testing = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
training = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")


# In[ ]:


#let us see the top rows of the training data set
training.head()


# In[ ]:


#let us see the bottom rows of the training data set
training.tail()


# In[ ]:


# how many rows and columns are there in the training data sets..?
training.shape


# In[ ]:


# let us see the summary of the training data set
training.info()


# In[ ]:


#let us check the descriptive stats
training.describe()


# ### What we learned.
# 
# We get the training set first and after one or two month we get the testing data set usually in companies,So training data set is all we have to play with.
# 1. There are 1460 rows and 81 columns, the SalePrice is our target feature(or column), it is very important to learn more about it.
# 2. Now most important thing, There is null values. we can clearly see there are many feaatures with null values, ex., columns = Alley,PoolQC.
#    But we need to find weather these features can give us some valuable information or not.
# 3. From the descriptive stats we can see some sudden jumps,so skewness and outliers are present.
# 

# ## Check the null values, numerical and catagorical features.

# In[ ]:


# let us check the number of catagorical and numerical features 
df = training.dtypes.reset_index()
df.columns = ['Count','Column Type']
df.groupby('Column Type').aggregate('count').reset_index()


# In[ ]:


# numerical features present in the training dataset
numeric_feats = training.select_dtypes(exclude=['object'])
numeric_feats.columns


# In[ ]:


# catagorical features present in the training dataset
categoric_feats = training.select_dtypes(include=['object'])
categoric_feats.columns


# In[ ]:


#missing data percent plot
total = training.isnull().sum().sort_values(ascending=False)
percent = (training.isnull().sum()/training.isnull().count()*100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.index.name ='Features'
missing_data.head(25)


# ### what we learned 
# 
# 1. There are 38 numerical features and 43 catagorical features. SalePrice is numeric and continues.
# 
# 2. There are 4 features with more than 50% null values and one with 47% percentage (usually 50% and more null value will not be usefull BUT be careful        sometime they provide some usefull information.

# ## Doing visualizations to get some insight of the data

# In[ ]:


# skewness in the training dataset
plt.figure(figsize = (12,8))
sns.distplot(training.skew(),color='blue',axlabel ='Skewness')
plt.show()


# In[ ]:


#kurtosis in the dataset
plt.figure(figsize = (12,8))
sns.distplot(training.kurt(),color='r',axlabel ='Kurtosis',norm_hist= False, kde = True,rug = False)
#plt.hist(train.kurt(),orientation = 'vertical',histtype = 'bar',label ='Kurtosis', color ='blue')
plt.show()


# In[ ]:


# correlation values between the features
correlation = numeric_feats.corr()
print(correlation['SalePrice'].sort_values(ascending = False),'\n')


# In[ ]:


#ploting the correlation of numeric features with sale price 
f , ax = plt.subplots(figsize = (14,12))
plt.title('Correlation of Numeric Features with Sale Price',y=1,size=16)
sns.heatmap(correlation,square = True,  vmax=0.8)


# In[ ]:


# zooming the correlation matrix 
k= 11
cols = correlation.nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(training[cols].values.T)
f , ax = plt.subplots(figsize = (14,12))
sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)


# In[ ]:


#ploting the pair plot 
sns.set()
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
sns.pairplot(training[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()


# In[ ]:




# Plot histogram and probability
fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(training['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(training['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.subplot(1,2,2)
res = stats.probplot(training['SalePrice'], plot=plt)
plt.suptitle('Before transformation')


# Apply transformation
training.SalePrice = np.log1p(training.SalePrice )
# New prediction
y_train = training.SalePrice.values
y_train_orig = training.SalePrice




# Plot histogram and probability after transformation
fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.distplot(training['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(training['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.subplot(1,2,2)
res = stats.probplot(training['SalePrice'], plot=plt)
plt.suptitle('After transformation')


# ## What we learned.
# 
# 1. The first graph shows that data is positively skewed(see the shape,you can see that the tail is extending to the right hand side). ML model work better for normaly distributed data(no skew). So we need transformation.
# 
#     There are different methods avilable : 
#     
#     (a)for right skewed data: log transformation for right skew 
#     
#     (b)left skewed data:  take reflection of left skew(then it will be a right skew) and do a log transform on that
#     
#     (c)Boxcox transformation(i am using this here)
# 2. Second graph shows the kurtosis(peakness of the data)
# 3. Third graph gives the heat map.
# 
# First I notice two features(TotalBsmtSF,1stFlrSF) have same colour this means multicollinearity is present.we need any one of this
#  
# Similarly(GarageCars,GarageArea) we need any one of these.
# 
# Another aspect I observed here is the 'SalePrice' correlations with 'GrLivArea', 'TotalBsmtSF', and 'OverallQual' , however we cannot exclude the fact     that rest of the features have some level of correlation to the SalePrice.
# 
# 4. Fourth graph(zoomed heat map):
# 
# 'OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'.
# 
# 'GarageCars' and 'GarageArea' are strongly correlated variables. It is because the number of cars that fit into the garage is a consequence of the garage area. 'GarageCars' and 'GarageArea' are like twin brothers. So it is hard to distinguish between the two. Therefore, we just need one of these variables in our analysis (we can keep 'GarageCars' since its correlation with 'SalePrice' is higher).
# 
# 'TotalBsmtSF' and '1stFloor' also seem to be twins. In this case let us keep 'TotalBsmtSF'
# 
# 'TotRmsAbvGrd' and 'GrLivArea', twins
# 
# 'YearBuilt' it appears like is slightly correlated with 'SalePrice'. This required more analysis to arrive at a conclusion may be do some time series analysis.
# 
# 5. pair plot: 
# 
# -One interesting observation is between 'TotalBsmtSF' and 'GrLiveArea'. In this figure we can see the dots drawing a linear line, which almost acts like a border. It totally makes sense that the majority of the dots stay below that line. Basement areas can be equal to the above ground living area, but it is not expected a basement area bigger than the above ground living area.
# 
# One more interesting observation is between 'SalePrice' and 'YearBuilt'. In the bottom of the 'dots cloud', we see what almost appears to be a exponential function.We can also see this same tendency in the upper limit of the 'dots cloud' 
# 
# Last observation is that prices are increasing faster now with respect to previous years.
# 
# 6. Target : Is log transformed 

# In[ ]:


#concat the train and test dataset into all_data
all_data = pd.concat((training,testing)).reset_index(drop=True)


# In[ ]:


#delete the 'SalePrice' column
ntrain = training.shape[0]
ntest = testing.shape[0]

all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# In[ ]:


#we will remove the least correlated features from the dataset to make faster and more accurate predictions
all_data.drop(columns={'Id','3SsnPorch','BedroomAbvGr','BsmtFinSF2','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','EnclosedPorch','KitchenAbvGr','LotArea','LowQualFinSF','MSSubClass','MiscVal','MoSold','OverallCond','PoolArea','ScreenPorch','YrSold'},inplace=True)


# In[ ]:


#it is advisable to remove the columns having high null values (considering the correlation of that variable)
all_data.drop(columns={'PoolQC','MiscFeature','Alley','Fence','FireplaceQu'},inplace=True)


# In[ ]:


all_data.head()


# ## Missing value imputation

# In[ ]:


#there may be no garage, so filling NaN with 0
all_data['GarageYrBlt']=all_data['GarageYrBlt'].fillna(0)


#no masonry veneer for some houses
all_data['MasVnrType']=all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea']=all_data['MasVnrArea'].fillna(0)


#NaN means there is no basement, so filling the null values with 'None'
for col in('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):
    all_data[col]=all_data[col].fillna('None')
    
#there is only one missing value, so filling it with mode
all_data['Electrical']=all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])


#no basement, so filling it with zero
for col in ('BsmtFinSF1','TotalBsmtSF'):
    all_data[col] = all_data[col].fillna(0)
    

#replacing null values with 'None' as there may be no garage in the house
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
    
    
#filling with the most occurring value
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


#mostly the dataset consists of same value in 'Utilities', so droping this feature
all_data = all_data.drop(['Utilities'], axis=1)


#NA for 'Functional' variable means typical (as per data description)
all_data["Functional"] = all_data["Functional"].fillna("Typ")


#filling it with most frequent value
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


#as there is only one missing value, we will impute it with the most occurring value
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

#imputing with mode
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


#there is no garage, so fill it with 0
for col in ('GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    

 #Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# In[ ]:


all_data.isnull().values.any()


# In[ ]:


#let's check for the skewed features


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# In[ ]:


#performing BoxCox transformation for the highly skewed features

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)


# In[ ]:


#dummy categorical features
all_data = pd.get_dummies(all_data,drop_first=True)
print(all_data.shape)


# In[ ]:


all_data.head()


# In[ ]:


#get the new train and test dataset
train = all_data[:ntrain]
test = all_data[ntrain:]


# In[ ]:


train.head()


# In[ ]:



test.head()


# In[ ]:


#splitting the data for train and test 

X_train, X_test, y_train, y_test = train_test_split(
                                     train, y_train,
                                     test_size=0.25,
                                     random_state=42
                                     )


# In[ ]:


#check the shape of all sets
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


linreg = LinearRegression()
parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}
grid_linreg = GridSearchCV(linreg, parameters_lin, verbose=1 , scoring = "r2")
grid_linreg.fit(X_train, y_train)

print("Best LinReg Model: " + str(grid_linreg.best_estimator_))
print("Best Score: " + str(grid_linreg.best_score_))


# In[ ]:


linreg = grid_linreg.best_estimator_
linreg.fit(X_train, y_train)
lin_pred = linreg.predict(X_test)
r2_lin = r2_score(y_test, lin_pred)
rmse_lin = np.sqrt(mean_squared_error(y_test, lin_pred))
print("R^2 Score: " + str(r2_lin))
print("RMSE Score: " + str(rmse_lin))


# In[ ]:


scores_lin = cross_val_score(linreg, X_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lin)))


# In[ ]:


lasso = Lasso()
parameters_lasso = {"fit_intercept" : [True, False], "normalize" : [True, False], "precompute" : [True, False], "copy_X" : [True, False]}
grid_lasso = GridSearchCV(lasso, parameters_lasso, verbose=1, scoring="r2")
grid_lasso.fit(X_train, y_train)

print("Best Lasso Model: " + str(grid_lasso.best_estimator_))
print("Best Score: " + str(grid_lasso.best_score_))


# In[ ]:


lasso = grid_lasso.best_estimator_
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
r2_lasso = r2_score(y_test, lasso_pred)
rmse_lasso = np.sqrt(mean_squared_error(y_test, lasso_pred))
print("R^2 Score: " + str(r2_lasso))
print("RMSE Score: " + str(rmse_lasso))


# In[ ]:


scores_lasso = cross_val_score(lasso, X_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lasso)))


# In[ ]:


ridge = Ridge()
parameters_ridge = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False], "solver" : ["auto"]}
grid_ridge = GridSearchCV(ridge, parameters_ridge, verbose=1, scoring="r2")
grid_ridge.fit(X_train, y_train)

print("Best Ridge Model: " + str(grid_ridge.best_estimator_))
print("Best Score: " + str(grid_ridge.best_score_))


# In[ ]:


ridge = grid_ridge.best_estimator_
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
r2_ridge = r2_score(y_test, ridge_pred)
rmse_ridge = np.sqrt(mean_squared_error(y_test, ridge_pred))
print("R^2 Score: " + str(r2_ridge))
print("RMSE Score: " + str(rmse_ridge))


# In[ ]:


scores_ridge = cross_val_score(ridge, X_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_ridge)))


# In[ ]:


dtr = DecisionTreeRegressor()
parameters_dtr = {"criterion" : ["mse", "friedman_mse", "mae"], "splitter" : ["best", "random"], "min_samples_split" : [2, 3, 5, 10], 
                  "max_features" : ["auto", "log2"]}
grid_dtr = GridSearchCV(dtr, parameters_dtr, verbose=1, scoring="r2")
grid_dtr.fit(X_train, y_train)

print("Best DecisionTreeRegressor Model: " + str(grid_dtr.best_estimator_))
print("Best Score: " + str(grid_dtr.best_score_))


# In[ ]:


dtr = grid_dtr.best_estimator_
dtr.fit(X_train, y_train)
dtr_pred = dtr.predict(X_test)
r2_dtr = r2_score(y_test, dtr_pred)
rmse_dtr = np.sqrt(mean_squared_error(y_test, dtr_pred))
print("R^2 Score: " + str(r2_dtr))
print("RMSE Score: " + str(rmse_dtr))


# In[ ]:


scores_dtr = cross_val_score(dtr, X_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_dtr)))


# In[ ]:


rf = RandomForestRegressor()
paremeters_rf = {"n_estimators" : [5, 10, 15, 20], "criterion" : ["mse" , "mae"], "min_samples_split" : [2, 3, 5, 10], 
                 "max_features" : ["auto", "log2"]}
grid_rf = GridSearchCV(rf, paremeters_rf, verbose=1, scoring="r2")
grid_rf.fit(X_train, y_train)

print("Best RandomForestRegressor Model: " + str(grid_rf.best_estimator_))
print("Best Score: " + str(grid_rf.best_score_))


# In[ ]:


rf = grid_rf.best_estimator_
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
r2_rf = r2_score(y_test, rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
print("R^2 Score: " + str(r2_rf))
print("RMSE Score: " + str(rmse_rf))


# In[ ]:


scores_rf = cross_val_score(rf, X_train, y_train, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_rf)))


# In[ ]:


model_performances = pd.DataFrame({
    "Model" : ["Linear Regression", "Ridge", "Lasso", "Decision Tree Regressor", "Random Forest Regressor"],
    "Best Score" : [grid_linreg.best_score_,  grid_ridge.best_score_, grid_lasso.best_score_, grid_dtr.best_score_, grid_rf.best_score_],
    "R Squared" : [str(r2_lin)[0:5], str(r2_ridge)[0:5], str(r2_lasso)[0:5], str(r2_dtr)[0:5], str(r2_rf)[0:5]],
    "RMSE" : [str(rmse_lin)[0:8], str(rmse_ridge)[0:8], str(rmse_lasso)[0:8], str(rmse_dtr)[0:8], str(rmse_rf)[0:8]]
})
model_performances.round(4)

print("Sorted by Best Score:")
model_performances.sort_values(by="Best Score", ascending=False)


# In[ ]:


print("Sorted by RMSE:")
model_performances.sort_values(by="RMSE", ascending=True)


# In[ ]:


ridge.fit(X_train, y_train)


# In[ ]:


submission_predictions = np.exp(ridge.predict(test))


# In[ ]:


submission_predictions.shape


# In[ ]:


submission = pd.DataFrame({
        "Id": testing["Id"],
        "SalePrice": submission_predictions
    })

submission.to_csv("prices.csv", index=False)
print(submission.shape)


# # please Upvote me, Thank you

# In[ ]:


Alhandu lillah

