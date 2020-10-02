#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, ElasticNetCV, Lasso
from sklearn.metrics import mean_squared_error, SCORERS
from sklearn.kernel_ridge import KernelRidge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
import warnings
from datetime import datetime


# In[ ]:


warnings.filterwarnings('ignore')
GD = False


# In[ ]:


ROOT = '/kaggle/input/house-prices-advanced-regression-techniques/'
TEST = os.path.join(ROOT, 'test.csv')
TRAIN = os.path.join(ROOT, 'train.csv')
df_test = pd.read_csv(TEST)
df_train = pd.read_csv(TRAIN)


# # 1. Context
# Based on the information available, we want to predict the price of a house based on a set of specific features. The answer to this question implies a regression model.
# 
# Since we are looking to use a linear regression model, we'll want to keep features that have a significant correlation with our target value (`SalesPrice`). Keeping this in mind, we'll need to evaluate the correlation of 3 types of variables:
# 1. Discrete and continuous variable (we'll be using pearson-r)
# 2. Binary variable (we'll be using a point-biseral correlation)
# 3. Categorical variable with more than 2 options (we'll use the [correlation ratio](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9))
# 
# We'll also want to exclude features correlated with each other (colinearity). Another important element we'll want to look at is the type of distribution our features have. Since we'll use a regression model we need to have our data normaly distributed - this may require to apply some transformation.

# # 2. Data Exploration
# ## 2.1 High Level Data Structure 
# We'll start by looking at our data from a hugh level. We'll try to understand:
# * shape of data
# * data type
# * missing values
# 
# We'll also seperate our dataset into target (y) and features (X). Finaly, if we find any missing values we'll work on either dropping the column or replacing null values.

# In[ ]:


rows, cols = df_train.shape
print(f'Training Dataset\n-------\ncolumns: {cols}\nrows: {rows}')
cat_cols = df_train.loc[:, df_train.columns != 'SalePrice'].select_dtypes(include=['object']).columns
num_cols = df_train.loc[:, df_train.columns != 'SalePrice'].select_dtypes(exclude=['object']).columns
print(f'categorical columns: {len(cat_cols)}\nnumeric columns: {len(num_cols)}\n\n=================\n')

rows, cols = df_test.shape
print(f'Test Dataset\n-------\ncolumns: {cols}\nrows: {rows}')
cat_cols = df_test.loc[:, df_test.columns != 'SalePrice'].select_dtypes(include=['object']).columns
num_cols = df_test.loc[:, df_test.columns != 'SalePrice'].select_dtypes(exclude=['object']).columns
print(f'categorical columns: {len(cat_cols)}\nnumeric columns: {len(num_cols)}')


# Our training dataset has 1,460 rows and 81 columns (80 features total). This is a pretty small dataset with a somewhat large amount of features. Out of our 80 columns we have:
# * 37 numeric columns
# * 43 categorical/string columns
# 
# This tells us we'll have to do some encoding on our categorical values to be able to take advantage of the full dataset

# In[ ]:


nulls = {}

for col in df_train.columns:
    nulls[col] = (1-(len(df_train[df_train[col].isna()][col]) / df_train.shape[0]))

labels = []
vals = []

for k, v in nulls.items():
    if v < 1.0:
        labels.append(k)
        vals.append(v)

_, ax = plt.subplots(figsize=(12,5))

sns.barplot(y=vals, x=labels, color='lightskyblue')
ax.set_xticklabels(labels=labels, rotation=45)
plt.title('% non-null values by columns')
ax.set_xlabel('columns')
ax.set_ylabel('%')
plt.show()


# Most of our columns have non-null data (18 out of 81). Among those 18 only 4 have an amount of non-null data that is very small (<20%). Base on this information, it is fair to drop those columns from our dataset.

# In[ ]:


to_drop = []

for k, v in nulls.items():
    if v < 0.6:
        to_drop.append(k)

# Let's use a copy of our dataframe so that we won't have to reload our entire dataset in case we need
# to do so (especially a good idea when we are working with very large dataset)
df_train_c = df_train.drop(to_drop, axis=1)

rows, cols = df_train_c.shape
print(f'columns: {cols}\nrows: {rows}')
cat_cols = df_train_c.loc[:, df_train_c.columns != 'SalePrice'].select_dtypes(include=['object']).columns
num_cols = df_train_c.loc[:, df_train_c.columns != 'SalePrice'].select_dtypes(exclude=['object']).columns
print(f'categorical columns: {len(cat_cols)}\nnumeric columns: {len(num_cols)}')


# Let's now fill in the missing values for our 14 columns with N/A values. We'll use the `SimpleImputer()` method using the most frequent value present in the column to replace null values.

# In[ ]:


si = SimpleImputer(strategy='most_frequent')

for k,v in nulls.items():
    if (v < 1) and (k not in to_drop):
        df_train_c[k] = si.fit_transform(df_train_c[k].values.reshape(-1,1))


# In[ ]:


df_train_c = df_train_c[df_train_c.GrLivArea < 4000]


# We now have 77 columns. The 4 columns we dropped were all categorical.
# Now that we have a somewhat cleaner set, we can start working on understanding the correlation of variables with `SalePrice`. First we'll seperate our dependent (y) and independent variables (X). We'll also remove the `Id` from our dataframe as we do not need this information right now.

# In[ ]:


X = df_train_c.loc[:, df_train_c.columns != 'SalePrice']
y = df_train_c.loc[:, df_train_c.columns == 'SalePrice']

df_train_ID = df_train.Id
df_test_ID = df_test.Id

X = X.loc[:, X.columns != 'Id']


# ## 2.1 Checking Data Skewness
# As we are going to perform a regression to predict our house sales price, we should make sure our numerical features are normaly distributed. If not, we should apply a transform before moving forward.
# 
# Let's first take a look at our target variable y.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(15,5))

sns.distplot(y, fit=stats.norm, ax=ax[0], kde=False)
stats.probplot(y.SalePrice,  plot=ax[1])
plt.show()
print(f'Fisher-Pearson coeficient of skewness: {stats.skew(y.SalePrice.values):.2f}')


# Our code gives us 3 output:
# * our data distribution with a normal fitted. This gives a way to see the actual distribution of our data
# * a QQ plot. This is use as a visual check to see if our data is normaly distributed. It sorts our values in ascending order (y-axis) and plot these againsta theorical quantiles (x-axis) from a normal distribution. A normaly distributed set will form a straigh line
# * Fisher-Pearson coefficient of skewness. A coeficient of 0 indicates no skenews while a positive coeficient indicates a right skewed distribution
# 
# We can visualy see that our target varibale is not normally distributed. This is confirmed when we compute the . We should apply a transformation. 
# 
# Let's now check if any of our numerical varibles are normaly distributed. We'll consider that any feature with an absolute skewness greater than 0.5 is skewed and will need to be transformed.
# We know that the following features are categorical variable that have been encoded so we should ignore them:
# * `MSSubClass`
# * `OverallQual` 
# * `OverallCond`

# In[ ]:


numerical_columns = X.loc[:, ~X.columns.isin(['MSSubClass', 'OverallQual', 'OverallCond', 'GarageYrBlt'])].select_dtypes(include=['int', 'float']).columns
sk = X[numerical_columns].apply(lambda x: stats.skew(x.dropna())).to_frame('Fisher-Pearson Coef')
skw_cols = list(sk[abs(sk['Fisher-Pearson Coef']) > 0.5].index)
sk[abs(sk['Fisher-Pearson Coef']) > 0.5]


# 24 of our numerical columns have a skewed distribution. We'll need to transform these in order to perform our regression. We'll use a Box Cox transformation. It is important to keep the lambda value constant in our transformation when using box cox.
# 
# A lambda value of 0 for the Box Cox transformation simply applies a log transformation.

# In[ ]:


lmbda = 0.0
X[skw_cols] = X[numerical_columns].loc[:, X[numerical_columns].columns.isin(skw_cols)].apply(lambda x: stats.boxcox(1+x, lmbda=lmbda))


# In[ ]:


y = y.apply(lambda x: stats.boxcox(1+x, lmbda=lmbda))


# We now have applied our transformation to our training data. Let's check the skewness of our data again to confirm we are now working with clean data.

# In[ ]:


sk['Fisher-Pearson Coef (After)'] = X[numerical_columns].apply(lambda x: stats.skew(x))
sk[sk.index.isin(skw_cols)]


# Not all of our columns have been successfuly transformed, though the majority of them now have a distribution with a coeficient of skewness close to 0.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(15,5))

sns.distplot(y, fit=stats.norm, ax=ax[0], kde=False)
stats.probplot(y.SalePrice,  plot=ax[1])
plt.show()
print(f'Fisher-Pearson coeficient of skewness: {stats.skew(y.SalePrice.values):,.2f}')


# The transformation for our target variable had a great effect. It is now almost normally distributed with a very low coeficient of skewness.

# ### 2.2 Selecting our features
# #### 2.2.1 Discrete Variables
# Let's start with our numerical values. This is the simpliest. `Pandas` offers a simple way to calculate pearson-r correlation, so this will be straightforward. Now we need to be award of a few element:
# * `MSSubClass`, `OverallQual`, `OverallCond`, are encoded categorical variable. We may need to re-encode with a code starting at 0
# * `MoSold` and `YrSold`, `YearBuilt`, `YearRemodAdd`, `GarageYrBlt` are discret variable and a spearman correlation may be more appropriate
# 
# Let's exclude those features and run our pearson correlation only against discrete variables.

# In[ ]:


X_disc = X.loc[:,~(X.columns.isin(['MoSold', 'YrSold', 'YearBuilt', 'YearRemodAdd', 'Id'])) &
                  (X.columns.isin(numerical_columns))]

X_disc['y'] = y

_, ax = plt.subplots(figsize=(25,15))

sns.heatmap(X_disc.corr(), annot=True, cbar=False, cmap='YlGnBu')
plt.show()


# We are now able to see the correlation of our discrete variables to our dependent variable `y`. We'll keep only variables with a moderate to strong correlation (i.e > |0.3|).

# In[ ]:


mask = (abs(X_disc.corr()['y'] >= 0.3))
corr_variables = X_disc.corr()['y'][mask]
corr_variables = list(corr_variables[corr_variables.index != 'y'].index)

corr_variables


# Now that we have selected our discrete features let's controle for colinearity in features. Features with strong correlation between each other will explain the same information in our model, hence we can get rid of one of the 2 variables.

# In[ ]:


_, ax = plt.subplots(figsize=(15,8))

sns.heatmap(X_disc.loc[:, corr_variables].corr(), annot=True, cbar=True, cmap='YlGnBu')
plt.show()


# In[ ]:


mask = ((abs(X_disc.loc[:, corr_variables].corr()) > 0.8) & 
        (X_disc.loc[:, corr_variables].corr() != 1.0))
cols = list(X_disc.loc[:, corr_variables].corr()[mask].dropna(how='all', axis=1).columns)

to_remove = []

for i in range(0,len(cols),2):
    to_remove.append(cols[i])
    
continous_features = list(set(corr_variables) - set(to_remove))
continous_features


# #### 2.2.2 Continuous Variables
# Now that we have our continuous numerical data, let's look at our discrete variables. One thing to note is that `YearRemodAdd` would be equal to `YearBuilt` if no remodeling has been done. Therefore, we'll engineer it into a `1`/`0` dichotomous variable where `1` indicates a remodeled house and `2` indicates a non remodeled house. 
# 
# We'll also add 3 new field in replacement to `YearBuilt`,`'YearRemodAdd`,`GarageYrBlt` capturing the rencency (as opposed to the year). It is fair to assume that what the relationship really capture is the timeframe between the the remodeling/construction and the sell date.

# In[ ]:


X['IsRemod'] = np.where(np.expm1(X[['YearBuilt']]).astype('int').YearBuilt == X.YearRemodAdd, 0, 1)
X['YrSinceBuilt'] = X.YrSold - X.YearBuilt
X['YrSinceRemod'] = X.YrSold - X.YearRemodAdd
X['YrSinceGarageYrBlt'] = X.YrSold - X.GarageYrBlt
X['HasMasVnr'] = np.where(X.MasVnrType == 'None',0,1)


# In[ ]:


# tmp = X[X['YrSinceRemod'] == 0]
# X =  X[X['YrSinceRemod'] != 0]
# tmp['YrSinceRemod'] = tmp.YrSinceRemod.replace(0,np.nan)
# X = X.append(tmp)


# In[ ]:


X_discrete = X.loc[:, X.columns.isin(['MoSold', 'YrSold', 'YrSinceBuilt', 'YrSinceRemod','YrSinceGarageYrBlt'])]
X_discrete['y'] = y

_, ax = plt.subplots(figsize=(15,8))

sns.heatmap(X_discrete.corr('spearman'), annot=True, cmap='YlGnBu')
plt.show()


# We can see 3 features that have a medium to high correlation with our dependent variable. We'll check for any colinearality between these 3 variables. The relationship tend to indicate that as the recency grows (was built or renovated farther in the past) we get a smaller price. Now, we can see a pretty strong colinerality between these 3 variables (especially between garage and construction). Hence we'll go ahead and only keep 2 of these 3 variables:
# * `YrSinceBuilt`
# * `YrSinceRemod`

# In[ ]:


mask = (((abs(X_discrete.corr('spearman')['y']) >= 0.3) &
       (X_discrete.corr('spearman')['y'] != 1.0)))
X_discrete_cols = list(X_discrete.corr('spearman')['y'][mask].index)
discrete_features = list(set(X_discrete_cols) - set(['YrSinceGarageYrBlt']))


# In[ ]:


X_num = X.loc[:, X.columns.isin(continous_features + discrete_features)]
X_num['y'] = y


# Now that we have our numerical features let's plot these variable against our target variable `SalePrice -` to have a visuale representation of the relationship.

# In[ ]:


sns.pairplot(x_vars=continous_features[:int(len(continous_features)/2)],
             y_vars=['y'],
            data=X_num,
            height=3.5)

sns.pairplot(x_vars=continous_features[int(len(continous_features)/2):],
             y_vars=['y'],
            data=X_num,
            height=3.5)

sns.pairplot(x_vars=discrete_features,
             y_vars=['y'],
            data=X_num,
            height=3.5)

plt.show()


# #### 2.2.3 Dichotomous Variables
# To analyse the relationship between our dichotomous variable and our discrete dependent variable `y` we'll use the point-biseral correlation. It is important to note that we'll consider only natural dichotomous variables. We have the following variables:
# * `CentralAir`
# * `IsRemod`

# In[ ]:


le = LabelEncoder()

X['CentralAir_enc']  = X[['CentralAir']].apply(lambda x: le.fit_transform(x.values))


# In[ ]:


r, p = stats.pointbiserialr(X.IsRemod.values, y.values.ravel())
print(f'IsRemod - r: {r} | p: {p}')
r, p = stats.pointbiserialr(X.CentralAir_enc.values, y.values.ravel())
print(f'CentralAir_enc - r: {r} | p: {p}')
r, p = stats.pointbiserialr(X.HasMasVnr.values, y.values.ravel())
print(f'MasVnr_enc - r: {r} | p: {p}')


# In[ ]:


dico = ['CentralAir_enc', 'HasMasVnr']


# Based on the standard we set to consider a relationship with our target variable, we can see that `CentralAir_enc` and `MasVnr_enc` both fit our requirement.
# #### 2.2.4 Categorical Variables
# Now let's move on to the categorical variables. We'll use the correlation ratio to measure the relationship between our categorical value and our target value. Correlation ratio range from 0-1 where 1 indicates the variance in our target value comes from differences between categories and where 0 indicates the differences in our target value comes from differences within our categories.
# 
# What we are interested is the variance between category (i.e. a value close to 1) as it indicates that belonging to a specific category influences the `SalePrice`

# In[ ]:


categoricals = list(X.loc[:,X.columns != 'CentralAir'].select_dtypes(include='object').columns)
categoricals = categoricals + ['MSSubClass', 'OverallQual', 'OverallCond']


# In[ ]:


X_categoricals = X[categoricals].apply(lambda x: le.fit_transform(x))
X_categoricals['y']  = y


# In[ ]:


corr = []


for col in tqdm(X_categoricals.columns):
    cat = X_categoricals[col].unique()
   
    y_avg = []
    n_cat = []
    for c in cat:
        y_avg.append(X_categoricals[X_categoricals[col] == c].y.mean())
        n_cat.append(len(X_categoricals[X_categoricals[col] == c]))
    
    y_total_avg = np.sum(np.multiply(y_avg,n_cat) / np.sum(n_cat))

    numerator = np.sum((np.multiply(n_cat,np.power(np.subtract(y_avg, y_total_avg),2))))
    denominator = np.sum(np.power(np.subtract(X_categoricals.y, y_total_avg),2))

    if denominator == 0:
        eta = 0.0
        corr.append((col, eta))
    else:
        eta = np.sqrt(numerator/denominator)
        corr.append((col, eta))
        
print(corr)


# In[ ]:


categoricals_columns = []

for el in corr:
    if el[1] >= 0.3:
        categoricals_columns.append(el[0])

categoricals_columns.pop(len(categoricals_columns)-1)
categoricals_columns


# In[ ]:


X_cat = X_categoricals[categoricals_columns]
X_cat['y'] = y


# In[ ]:


sns.pairplot(x_vars=categoricals_columns[:int(len(categoricals_columns)/2)],
             y_vars=['y'],
            data=X_cat,
            height=3.5)
sns.pairplot(x_vars=categoricals_columns[int(len(categoricals_columns)/2):],
             y_vars=['y'],
            data=X_cat,
            height=3.5)
plt.show()


# ## 3. Training & Testing our Model
# Let's first recap. what we have done so far:
# 1. *descriptive analytics*: we looked at the shape of our data, the missing values as well as the data types we are working with. We also did some very light data engineering and data transformation
# 2. *Selected our features*: after looking at the data we selected our features based on their correlation with our target variable. We split the feature selection based on the type of variables we were working with.
# 
# It is now time to train and test our model. We'll first define a baseline so that we can see how well our model is performing. For our model, I have chosen to use a stacked ensemble model using 4 submodels:
# * OLS Regression
# * Ridge Regression
# * ElasticNet Regression
# * GB Regression
# * XGB Regression
# The choice of uing a stacked model was driven by notebooks from other users and an interest in learning this technique.
# 
# ### 3.1 Bookkeeping
# Let's first do some bookkeeping work. First we'll combine all of our features, then split our model into a train, a test and a validation set and finaly get our baseline score. We'll use an untune single XGB Regressor model as a baseline.

# In[ ]:


features = categoricals_columns + continous_features + discrete_features + dico
X = X[features]
y = y
X[categoricals_columns] = X[categoricals_columns].apply(lambda x: le.fit_transform(x))
X_train, X_test, y_train, y_test = train_test_split(X.loc[:,X.columns != 'y'], y, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.3)


# Now that we have our sets, let's define our cross validation function that will be used to measure the performance of our model when performing some tunning.

# In[ ]:


n_fold = 5

def rmseModel(m):
    kf = KFold(n_splits=n_fold, random_state=0, shuffle=True).get_n_splits()
    rmse = np.sqrt(-cross_val_score(m, X, y, scoring='neg_mean_squared_error', cv=kf))
    return rmse


# In[ ]:


XGBBaselie = XGBRegressor(objective='reg:squarederror')
XGBBaselie.fit(X_test, y_test)
pred = XGBBaselie.predict(X_val)

rmseBaseline = np.sqrt(mean_squared_error(pred, y_val.values))
print(f'Baseline RMSE: {rmseBaseline}')


# ### 3.2 Model Definition
# We now have our baseline score and the approach we want to use to tackle this problem. Let's now train our models, implement our stacked ensemble and evaluate its performance against our baseline.

# #### 3.2.1 OLS Regression

# In[ ]:


ols_reg = LinearRegression()
ols_rge_scores = rmseModel(ols_reg)
print(f'OLS Reg RMSE, mean: {np.mean(ols_rge_scores)}, stdv: {np.std(ols_rge_scores)}')


# #### 3.2.2 Ridge Regression

# In[ ]:


if GD:
    print('Running Grid Search for model tunning')
    params = {'alpha': [0.1,0.3,0.5,0.7,0.9],
             'solver': ['auto', 'svd', 'cholesky', 'lsqr']}
    ridge_reg = Ridge()
    gs = GridSearchCV(ridge_reg, params, cv=5)
    gsf = gs.fit(X_train, y_train).best_params_
else:
    gsf = {'alpha': 0.9, 'solver': 'auto'}

ridge_reg = Ridge(**gsf)
ridge_reg_scores = rmseModel(ridge_reg)
print(f'Ridge Reg RMSE, mean: {np.mean(ridge_reg_scores)}, stdv: {np.std(ridge_reg_scores)}')


# #### 3.2.3 ElasticNet

# In[ ]:


if GD:
    print('Running Grid Search for model tunning')
    params = {'l1_ratio': [.1, .5, .7, .9, .92, .95, .99, 1],
             'n_alphas': [10,15,50, 100],
              'normalize': [True, False],
              'max_iter': [5,10,50,100],
              'tol': [0.001, 0.0001, 0.00001]
                }
    el_reg = ElasticNetCV()
    gs = GridSearchCV(el_reg, params, cv=5, n_jobs=-1, verbose=1)
    gsf = gs.fit(X_train, y_train).best_params_
else:
    gsf = {'l1_ratio': 0.9,
         'max_iter': 50,
         'n_alphas': 50,
         'normalize': True,
         'tol': 0.0001}

    
el_reg = ElasticNetCV(**gsf)
el_reg_scores = rmseModel(el_reg)
print(f'Elastic Net Reg RMSE, mean: {np.mean(el_reg_scores)}, stdv: {np.std(el_reg_scores)}')


# #### 3.2.4 Gradient Boost Regression
# We followed the approach laid out by [Aarshay Jain](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/) on [analyticsvidhya.com](http://analyticsvidhya.com) to tune our GB model.

# In[ ]:


if GD:
    print('Running Grid Search for model tunning')
    params = {'min_samples_split': [80],
             'min_samples_leaf': [25],
             'max_depth':[9],
            'max_features': [4],
             'subsample': [0.8],
             'n_estimators': [2500],
             'learning_rate': [0.005],
             'subsample':[0.87]}
    GB = GradientBoostingRegressor()
    gs = GridSearchCV(GB, param_grid=params, cv=5, n_jobs=-1, verbose=1)
    gsf = gs.fit(X_train, y_train).best_params_
else:
    gsf = {'learning_rate': 0.005,
         'max_depth': 9,
         'max_features': 4,
         'min_samples_leaf': 25,
         'min_samples_split': 80,
         'n_estimators': 2500,
         'subsample': 0.87}

GB_reg = GradientBoostingRegressor(**gsf)
GB_reg_scores = rmseModel(GB_reg)
print(f'GB Reg, mean: {np.mean(GB_reg_scores)}, stdv: {np.std(GB_reg_scores)}')


# #### 3.2.5 Extrem Gradient Boost Regression
# We followed the approach laid out by [Aarshay Jain](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/) on [analyticsvidhya.com](http://analyticsvidhya.com) to tune our XGB model.

# In[ ]:


if GD:
    print('Running Grid Search for model tunning')
    params = {'max_depth ': [1],
              'min_child_weight': [2],
              'gamma ': [0.0],
              'subsample':[0.7],
              'reg_alpha':[1e-5, 1e-4, 1e-6],
              'colsample_bytree': [0.87],
              'scale_pos_weight':[1],
                }

    xgb_reg = XGBRegressor()
    gs = GridSearchCV(xgb_reg, params, cv=5, n_jobs=-1, verbose=1)
    gsf = gs.fit(X_train, y_train).best_params_
else:
    gsf = {'colsample_bytree': 0.87,
         'gamma ': 0.0,
         'max_depth ': 1,
         'min_child_weight': 2,
         'reg_alpha': 1e-06,
         'scale_pos_weight': 1,
         'subsample': 0.7}
    
xgb_reg = XGBRegressor(**gsf, objective='reg:squarederror', nthread=4, learning_rate=0.005, n_estimators=10000)
xgb_reg_scores = rmseModel(xgb_reg)
print(f'XGB Reg, mean: {np.mean(xgb_reg_scores)}, stdv: {np.std(xgb_reg_scores)}')


# It is encouraging to see that all of tuned models perform better than our baseline - with our XGB model performing the best, though not as stable based on the standard deviation. We'll now build and test our stacked ensemble model and test how it performs.

# ### 3.3 Building and Testing our Ensemble Model
# #### 3.3.1 How is our model going to work?
# Our approach to building our stack ensemble will be to use the output from 4 of our model (OLS, Ridge, Elastic Net, and GB) as our input for an ensemble regression model. We'll then combine the prediction from this model with the one of the XGB (allocating appropriate weights) to get our final prediction.

# In[ ]:


olsM = ols_reg.fit(X_train, y_train)
elM = el_reg.fit(X_train, y_train)
RidgeM = ridge_reg.fit(X_train, y_train)
GBregM = GB_reg.fit(X_train, y_train)
XGBoostM = xgb_reg.fit(X_train, y_train)


# In[ ]:


ensembleOutput = np.hstack((olsM.predict(X_test), RidgeM.predict(X_test), elM.predict(X_test).reshape(-1,1), GBregM.predict(X_test).reshape(-1,1)))
stackedReg = LinearRegression()
sackedM = stackedReg.fit(ensembleOutput, y_test)


# In[ ]:


valEnsembleOutput = np.hstack((olsM.predict(X_val), RidgeM.predict(X_val), elM.predict(X_val).reshape(-1,1),GBregM.predict(X_val).reshape(-1,1)))
stackedPred = sackedM.predict(valEnsembleOutput)


# In[ ]:


pred = (np.expm1(stackedPred).reshape(1,-1)[0]*0.55 +np.expm1(XGBoostM.predict(X_val))*0.45)

rmse_test = np.sqrt(mean_squared_error(np.log(pred), y_val.values))
print(f'rmse for test data: {rmse_test}')


# # 4. Submitting our Predictions
# ## 4.1 Transforming our data

# In[ ]:


df_test['IsRemod'] = np.where(df_test.YearBuilt == df_test.YearRemodAdd, 0, 1)
df_test['YrSinceBuilt'] = df_test.YrSold - df_test.YearBuilt
df_test['YrSinceRemod'] = df_test.YrSold - df_test.YearRemodAdd
df_test['YrSinceGarageYrBlt'] = df_test.YrSold - df_test.GarageYrBlt
df_test['HasMasVnr'] = np.where(df_test.MasVnrType == 'None',0,1)
df_test['CentralAir_enc']  = df_test[['CentralAir']].apply(lambda x: le.fit_transform(x.values))

dfPred = df_test[features]


# In[ ]:


nulls = {}

for col in dfPred.columns:
    nulls[col] = (1-(len(dfPred[dfPred[col].isna()][col]) / dfPred.shape[0]))
    
for k, v in nulls.items():
    if v < 1.0:
        dfPred[k] = si.fit_transform(dfPred[k].values.reshape(-1,1))
        
dfPred[list(set(skw_cols).intersection(set(dfPred.columns)))] = dfPred[list(set(skw_cols).intersection(set(dfPred.columns)))].                                                                                apply(lambda x: stats.boxcox(1+x, lmbda=lmbda))

dfPred[categoricals_columns] = dfPred[categoricals_columns].apply(lambda x: le.fit_transform(x))


# ## 4.2 Making our Prediction

# In[ ]:


outputPred = np.hstack((olsM.predict(dfPred), RidgeM.predict(dfPred), elM.predict(dfPred).reshape(-1,1), GBregM.predict(dfPred).reshape(-1,1)))
stackedPred = sackedM.predict(outputPred)
finalPred = (np.expm1(stackedPred).reshape(1,-1)[0]*0.55 +np.expm1(XGBoostM.predict(dfPred))*0.45)


# ## 4.3 Submitting our File

# In[ ]:


dff = pd.DataFrame({
    'Id': df_test.Id,
    'SalePrice':finalPred
})

dff.to_csv(f"submission_{datetime.today().strftime('%Y%m%d')}.csv", index=False)


# ## 4.4 Results
# With this solution we rank 2,334 with an RMSE of 0.13507. This solution places us in the 54th percentile, which is slightly better than the average submission. Feel free to add any comments below on suggested ways to improve the existing model.
