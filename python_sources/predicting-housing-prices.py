#!/usr/bin/env python
# coding: utf-8

# # Predicting Housing Prices

# ## Loading Libraries

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from scipy.stats import probplot
from scipy.stats import skew

import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import rcParams

get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 8, 6
sb.set()


# ## Reading in Data

# In[ ]:


# Read in data
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print(train.shape)
print(test.shape)

train.head()


# ## Examine the Available Feautres
# 
# First let's see what variables are best correlated with Sale Price,

# In[ ]:


corr = train.corr()
plt.figure(figsize = (16, 12))
sb.heatmap(corr, linewidths = 0.5, fmt = '.2f', center = 1)
plt.show()


# Look at correlation with SalePrice: OveralQual, TotalBsmtSF, GrLivArea, FullBath, and GarageCars stand out with the strongest correlations.
# 
# ### Check for Linear Relationship

# Linear relationships are exhibited, but let's consider the distribution of SalePrice and then check back in.

# In[ ]:


sb.pairplot(train[['SalePrice','OverallQual', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'GarageCars']])
plt.show()


# ### Is Sale Price Normally Distributed?

# In[ ]:


y_train = train['SalePrice']
sb.distplot(y_train)
plt.xlabel('Sale Price')
plt.ylabel('Count')
plt.title('Distribution of Sale Price')
plt.show()

probplot(y_train, plot = plt)
plt.show()


# SalePrice is quite right-skewed and does not reflect a normal distribution. In the normal probability plot, the values do not follow the diagonal line. In its current state, SalePrice does not satisfy the requirements for linear regression. We can apply a log transformation to fix this.

# In[ ]:


train['SalePrice'] = np.log1p(train['SalePrice'])
y_train = train['SalePrice']
sb.distplot(y_train)
plt.xlabel('Sale Price')
plt.ylabel('Count')
plt.title('Distribution of Sale Price')
plt.show()

probplot(y_train, plot = plt)
plt.show()


# Now let's see if SalePrice and our chosen variables exhibit linearity again.

# In[ ]:


plt.scatter(train['OverallQual'], train['SalePrice'])
plt.title('OveralQual vs. SalePrice')
plt.show()

plt.scatter(train['TotalBsmtSF'], train['SalePrice'])
plt.title('TotalBsmtSF vs. SalePrice')

plt.show()

plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.title('GrLivArea vs. SalePrice')
plt.show()

plt.scatter(train['FullBath'], train['SalePrice'])
plt.title('FullBath vs. SalePrice')
plt.show()

plt.scatter(train['GarageCars'], train['SalePrice'])
plt.title('GarageCars vs. SalePrice')
plt.show()


# Linearity checks out.

# ## Pre-Processing Data
# 
# Combine the train and test datasets' features to universally apply any transformations.

# In[ ]:


train_features = train.drop(['SalePrice'], axis = 1)
test_features = test

features = pd.concat([train_features, test_features]).reset_index(drop = True)
features.shape


# ### Transforming Skewed Variables
# 
# We should check the normality of all the numeric variables.

# In[ ]:


# Selecting numeric variables with skewness > 0.5
numvars = features.select_dtypes(include = ['int64', 'float64', 'int32']).columns
numvars_skew = pd.DataFrame(features[numvars].skew(), columns = ['Skew'])
numvars_skew = numvars_skew[numvars_skew['Skew'] > 0.5]

# Applying log transformation to skewed variables
skewed = features[numvars_skew.index]
unskewed = np.log1p(skewed)

# Replacing in dataset
features[skewed.columns] = unskewed


# In[ ]:


fig = plt.figure(figsize = (16, 12))
ax = fig.add_subplot()

sb.boxplot(data=features[skewed.columns] , orient="h")
plt.show()


# Fixed well enough so that variables have more normal distributions.

# ### Handling Missing Values

# In[ ]:


# Sum of missing values
print("There are {} missing values in the features dataset.".format(features.isnull().sum().sum()))


# Let's get a summary of all the columns that have missing values, and how many there are.

# In[ ]:


# Summmary of columns with missing values
missing = features.columns[features.isnull().any()]
features[missing].isnull().sum().to_frame()


# I will later perform imputation some of these columns, but I don't want to keep columns with more than 33 percent of the data missing, which is ~1000 rows. Taking a look at those columns, it seems they are also not so important that I should argue to keep them.

# In[ ]:


# Remove columns with more than 33 percent missing values: > 1000
remove_cols = features.columns[features.isnull().sum() > 1000]
print('The following list contains columns in the train dataset which have more than 33 percent missing values: \n{}.'.format(remove_cols))

# Drop those columns from features
features = features.drop(remove_cols, axis = 1)

print('I will remove those columns. \nThere are now {} missing values in the features dataset.'.format(features.isnull().sum().sum()))


# ### Categorical Variables
# 
# Now, let's do a simple encoding of the categorical variables. I want to check the NA values here, too.

# In[ ]:


# List of train columns that are objects
objs = features.select_dtypes(include = ['object']).columns

# List of train columns that are objects with missing values
missing = features[objs].columns[features[objs].isnull().any()]
features[missing].isnull().sum().to_frame()


# I will do a mode imputation on these null values, filling them with the highest frequency value from the column.

# In[ ]:


# Fill train columns that are objects that have missing values with mode
features[missing] = features[missing].fillna(features.mode().iloc[0])
print("There are now {} missing values among the categorical variables.".format(features[objs].isnull().any().sum()))


# We can continue with our encoding process. I will just do label encoding because there are quite a few categorical values to manage, and one-hot encoding will get haphazard.

# In[ ]:


features[objs] = features[objs].apply(LabelEncoder().fit_transform)
features.info()


# We now have our categorical variables as integers and with no missing values. Let's see how many integer missing values are left over.

# In[ ]:


missing = features.columns[features.isnull().any()]
features[missing].isnull().sum().to_frame()


# Seems like we did a pretty solid job getting rid of many of the missing values. Let's do imputation on the rest now.
# 
# ### Imputation
# 
# We will replace the missing values with the mean value along each column. Since the encoded categorical variables no longer have missing values, we don't have to worry about subsetting that.

# In[ ]:


imputer = SimpleImputer()
imputed_features = pd.DataFrame(imputer.fit_transform(features))
imputed_features.columns = features.columns
features = imputed_features

# Drop the ID columns
features = features.drop('Id', axis = 1)

features.head()


# In[ ]:


print("There are now {} missing values in the features dataset.".format(features.isnull().any().sum()))


# ## Modelling
# 
# Let's look at the data we are working with and properly assign X_train, y_train, X_test.

# In[ ]:


features.shape


# In[ ]:


y_train = train['SalePrice']
X_train = features.iloc[:len(y_train), :]
X_test = features.iloc[len(y_train):, :]

X_train.shape, y_train.shape, X_test.shape


# ### Load Libraries

# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

# ignore warnings
import warnings as wrn
wrn.filterwarnings('ignore', category = DeprecationWarning) 
wrn.filterwarnings('ignore', category = FutureWarning) 
wrn.filterwarnings('ignore', category = UserWarning) 


# ### Initialize Models
# 
# We will be looking at a multitude of models and then examining their scores. From there, we can blend the models.
# 
# I will initialize all models with default values, except the seed.

# In[ ]:


seed = 1

linear = LinearRegression()
elastic = ElasticNet(random_state = seed)
ridge = Ridge(random_state = seed)
lasso = Lasso(random_state = seed)
kernel = KernelRidge()
r_forest = RandomForestRegressor(random_state = seed)
g_boost = GradientBoostingRegressor(random_state = seed)
svr = SVR()
knn = KNeighborsRegressor()
lgbm = LGBMRegressor(random_state = seed)
xgb = XGBRegressor(random_state = seed)


# ### K-Fold Cross Validation
# 
# Let's use k-fold cross validation to evaluate model performance. We will show the cross validation root mean squared error.

# In[ ]:


kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)

def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring = "neg_mean_squared_error", cv = kfold))
    rmse = np.round(rmse, 6)
    return(rmse)


# In[ ]:


scores = {}
scores['linear'] = cv_rmse(linear).mean()
scores['elastic'] = cv_rmse(elastic).mean()
scores['ridge'] = cv_rmse(ridge).mean()
scores['lasso'] = cv_rmse(lasso).mean()
scores['kernel'] = cv_rmse(kernel).mean()
scores['r_forest'] = cv_rmse(r_forest).mean()
scores['g_boost'] = cv_rmse(g_boost).mean()
scores['svr'] = cv_rmse(svr).mean()
scores['knn'] = cv_rmse(knn).mean()
scores['lgbm'] = cv_rmse(lgbm).mean()
scores['xgb'] = cv_rmse(xgb).mean()


# In[ ]:


plt.scatter(scores.keys(), scores.values())
plt.xticks(rotation = 45)
plt.title('10-Fold Cross Validation Scores')
plt.ylabel('RMSE')
plt.xlabel('Model')
plt.show()


# A lower RMSE means the model has performed better. 

# ### Fitting Models and Evaluating Predictions
# 
# Let's go ahead and train the models. 
# 
# We will use Root Mean Squared Log Error as an evaluation metric. RMSLE is more robust to the effect of outliers. We also use it because RMSLE incurs a larger penalty from the underestimation of the value. If we think about this relative to the business standpoint of assessing house prices, underestimating a house price is less acceptable than overestimating a house price -- whether that be because we hypothetically work for a real estate company selling these houses, or some entity looking to acquire these houses. It is better to overestimate.

# In[ ]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[ ]:


linear_model = linear.fit(X_train, y_train)
linear_pred = linear_model.predict(X_train)
linear_error = rmsle(y_train, linear_pred)

elastic_model = elastic.fit(X_train, y_train)
elastic_pred = elastic_model.predict(X_train)
elastic_error = rmsle(y_train, elastic_pred)

ridge_model = ridge.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_train)
ridge_error = rmsle(y_train, ridge_pred)

lasso_model = lasso.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_train)
lasso_error = rmsle(y_train, lasso_pred)

kernel_model = kernel.fit(X_train, y_train)
kernel_pred = kernel_model.predict(X_train)
kernel_error = rmsle(y_train, kernel_pred)

r_forest_model = r_forest.fit(X_train, y_train)
r_forest_pred = r_forest_model.predict(X_train)
r_forest_error = rmsle(y_train, r_forest_pred)

g_boost_model = g_boost.fit(X_train, y_train)
g_boost_pred = g_boost_model.predict(X_train)
g_boost_error = rmsle(y_train, g_boost_pred)

svr_model = svr.fit(X_train, y_train)
svr_pred = svr_model.predict(X_train)
svr_error = rmsle(y_train, svr_pred)

knn_model = knn.fit(X_train, y_train)
knn_pred = knn_model.predict(X_train)
knn_error = rmsle(y_train, knn_pred)

lgbm_model = lgbm.fit(X_train, y_train)
lgbm_pred = lgbm_model.predict(X_train)
lgbm_error = rmsle(y_train, lgbm_pred)

xgb_model = xgb.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_train)
xgb_error = rmsle(y_train, xgb_pred)


# We can see the RMSLE scores for the models.

# In[ ]:


print("Linear Model RMSLE:", linear_error)
print("Elastic Net Model RMSLE:", elastic_error)
print("Ridge Model RMSLE:", ridge_error)
print("LASSO Model RMSLE:", lasso_error)
print("Kernel Ridge Model RMSLE:", kernel_error)
print("Random Forest Model RMSLE:", r_forest_error)
print("Gradient Boosting Model RMSLE:", g_boost_error)
print("Suppor Vector Regression Model RMSLE:", svr_error)
print("K-Nearest Neighbors Model RMSLE:", knn_error)
print("LightGBM RMSLE:", lgbm_error)
print("XGBoost Model RMSLE:", xgb_error)

model_names = ['linear', 'elastic', 'ridge', 'lasso', 'kernel', 'r_forest', 'g_boost', 'svr', 'knn', 'lgbm', 'xgb']
rmsle_scores = [linear_error, elastic_error, ridge_error, lasso_error, kernel_error, r_forest_error, 
                              g_boost_error, svr_error, knn_error, lgbm_error, xgb_error]

plt.scatter(model_names, rmsle_scores)
plt.xticks(rotation = 45)
plt.title('Fitted Model RMSLE Scores')
plt.ylabel('RMSLE')
plt.xlabel('Model')
plt.show()


# Here we have all our models. Let's blend a few.

# In[ ]:


blended_pred = (r_forest_pred*0.25 + g_boost_pred*0.15 + svr_pred*0.15 + lgbm_pred*0.30 + xgb_pred*0.15)
blended_error = rmsle(y_train, blended_pred)
print("Blended Model RMSLE:", blended_error)


# In[ ]:


model_names = ['linear', 'elastic', 'ridge', 'lasso', 'kernel', 'r_forest', 
               'g_boost', 'svr', 'knn', 'lgbm', 'xgb', 'blended']
rmsle_scores = [linear_error, elastic_error, ridge_error, lasso_error, kernel_error, r_forest_error, 
                              g_boost_error, svr_error, knn_error, lgbm_error, xgb_error, blended_error]
plt.scatter(model_names, rmsle_scores)
plt.xticks(rotation = 45)
plt.title('Fitted Model RMSLE Scores')
plt.ylabel('RMSLE')
plt.xlabel('Model')
plt.show()


# For our final predictions, I am going to use our blended model.
# 
# ## Submission

# In[ ]:


r_forest_pred = r_forest_model.predict(X_test)
g_boost_pred = r_forest_model.predict(X_test)
svr_pred = r_forest_model.predict(X_test)
lgbm_pred = r_forest_model.predict(X_test)
xgb_pred = r_forest_model.predict(X_test)

blended_pred = (r_forest_pred*0.25 + g_boost_pred*0.15 + svr_pred*0.15 + lgbm_pred*0.30 + xgb_pred*0.15)

predictions = np.expm1(blended_pred)
submission = pd.DataFrame()
submission['Id'] = test['Id']
submission['SalePrice'] = predictions
submission.to_csv('submission.csv', index = False)


# In[ ]:


submission.head()


# It would be better to optimize the various parameters for the models we used. I might revisit that and some feature engineering at a later date. As of now, these steps have placed us in the top 67% of the competition.
# 
# Hope this was an enjoyable read.
# 
# Thank you!

# ## Appendix

# ### Models
# 
# __Linear Regression:__ A linear approach to modeling the relationship between target and one or more predictors. It attempts to minimize the sum of error squared.
# 
# __Ridge Regression:__ Decreases the model complexity by shrinking variable effects. Uses L2 regularization to add a penalty to the ordinary least squares (OLS) equation. Predictors with minor contribution have their ceofficients shrunk close to 0. This leads to lower variance and lower error value.
# 
# __LASSO Regression:__ Shrinks regression coefficients toward 0 by penalizing the model with the L1 norm. Forces some coefficient estimates to equal 0, removing some predictors from the model and reducing complexity.
# 
# __Elastic Net Regression:__ Regression model penalized with both the L1 and L2 norm. Shrinks some coefficients and sets some to 0.
# 
# __Kernel Ridge Regression:__ Combines Ridge Regression with kernel trick. 
# 
# __Random Forest Regression:__ Ensemble of different regression trees. Constructs many decision trees at training time and outputs the class that is the mean prediction of the individual trees.
# 
# __Gradient Boosting Regression:__ Boosting is an ensemble technique in which he predictors are made sequentially. Gradient boosting produces a prediction model in the form of an ensemble of weak prediction models, like decision trees.
# 
# __Decision Tree Regression:__ Uses binary rules to calculate a target value. Breaks down dataset into smaller and smaller subsets that contain instances with similar values, finally resulting in a tree wtih decision nodes and leaf nodes. 
# 
# __Super Vector Regression:__ Principles of SVM applied to regression. Minimizes error, individualizing a hyperplane which maximizes the margin, and tolerating a part of error.
# 
# __K-Nearest Neighbors Regression:__ Principles of KNN applied to regression. Uses the average of nearest data points to predict target.
