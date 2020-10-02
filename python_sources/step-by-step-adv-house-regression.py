#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries and Input Files

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


house_desc = open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt','r')
for i in house_desc:
    print(i)


# Numerical but categorical columns are: MSSubClass, OverallQual, OverallCond, MoSold, YrSold

# Lets have a look on the data.

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# Well, both train and test sets both contains so much columns that it is very difficult to read them at once. So, I will seperate them into Numerical and Categorical columns.

# But before that, I will drop 'Id' column from both training and test set.

# In[ ]:


train_data.drop(['Id'], axis = 1, inplace = True)
test_data.drop(['Id'], axis = 1, inplace = True)


# A quick view on training data

# In[ ]:


train_data.describe()


# ## Exploration Analysis

# Looking at the shapes of both the datasets

# In[ ]:


print(train_data.shape)
print(test_data.shape)


# Heatmap helps a lot in knowing the correlation between columns of a dataset.

# In[ ]:


plt.figure(figsize = (16,9))
corr_train_data = train_data.corr()
sns.heatmap(corr_train_data)
plt.title('Correleation Between Numerical Columns')
print("Top 20 Numeric Columns which are highly correlated to SalePrice are:")
print(corr_train_data.nlargest(21, 'SalePrice')['SalePrice'])
plt.show()


# After the following cell, I will have columns seperated as Numerical and Categorical Columns. 

# In[ ]:


cat_cols = [x for x in train_data.columns if train_data[x].dtype == 'object']
num_cols = [x for x in train_data.columns if train_data[x].dtype != 'object']


# In[ ]:


print('Number of Categorical Columns:',len(cat_cols))
print('Number of Numerical Columns:',len(num_cols))


# num_cols contain 'SalePrice' as well but that is our target so I will remove it for now for this list.

# In[ ]:


num_cols.remove('SalePrice')


# Now, as I seperated out categorical and numerical columns(very annoying to type those big names so I will call them as cat and num respectively), lets see the plot of num columns with the target column that is 'SalePrice'.

# In[ ]:


plt.figure(figsize = (16,10))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.scatter(train_data[num_cols[i]], train_data['SalePrice'])
    plt.title(num_cols[i])
plt.tight_layout()


# There are many columns which have outliers so I need to take care of them. Lets check the columns which have highest correlation with 'SalePrice'.

# In[ ]:


sns.jointplot(train_data['GrLivArea'], train_data['SalePrice'], kind = 'reg')
sns.jointplot(train_data['GarageArea'], train_data['SalePrice'], kind = 'reg')


# In[ ]:


train_data = train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<250000)].index).reset_index(drop=True)
train_data = train_data.drop(train_data[(train_data['GarageArea']>1150) & (train_data['SalePrice']<200000)].index).reset_index(drop=True)
sns.jointplot(train_data['GrLivArea'], train_data['SalePrice'], kind = 'reg')
sns.jointplot(train_data['GarageArea'], train_data['SalePrice'], kind = 'reg')


# So, I have removed 5 outliers from both the above columns. Lets try out the same for other columns which are in the list of top 20 corrrelated columns.

# In[ ]:


sns.jointplot(train_data['TotalBsmtSF'], train_data['SalePrice'], kind = 'reg')
sns.jointplot(train_data['1stFlrSF'], train_data['SalePrice'], kind = 'reg')


# 1 outlier from each above columns.

# In[ ]:


train_data = train_data.drop(train_data[(train_data['TotalBsmtSF']>6000) & (train_data['SalePrice']<200000)].index).reset_index(drop=True)
train_data = train_data.drop(train_data[(train_data['1stFlrSF']>4000) & (train_data['SalePrice']<200000)].index).reset_index(drop=True)
sns.jointplot(train_data['TotalBsmtSF'], train_data['SalePrice'], kind = 'reg')
sns.jointplot(train_data['1stFlrSF'], train_data['SalePrice'], kind = 'reg')


# 2 outliers from above columns. Lets look at more columns.

# In[ ]:


sns.jointplot(train_data['YearBuilt'], train_data['SalePrice'], kind = 'reg')
sns.jointplot(train_data['YearRemodAdd'], train_data['SalePrice'], kind = 'reg')
sns.jointplot(train_data['GarageYrBlt'], train_data['SalePrice'], kind = 'reg')


# These above columns do not seem to have any outliers. Lets try some other columns as well.

# In[ ]:


sns.jointplot(train_data['MasVnrArea'], train_data['SalePrice'], kind = 'reg')
sns.jointplot(train_data['BsmtFinSF1'], train_data['SalePrice'], kind = 'reg')


# I dont think there is need to clean those columns as well.

# In[ ]:


sns.jointplot(train_data['LotFrontage'], train_data['SalePrice'], kind = 'reg')
sns.jointplot(train_data['WoodDeckSF'], train_data['SalePrice'], kind = 'reg')
sns.jointplot(train_data['2ndFlrSF'], train_data['SalePrice'], kind = 'reg')


# Looks like 1 outlier in LotFrontage and all others are looking good.

# In[ ]:


sns.jointplot(train_data['OpenPorchSF'], train_data['SalePrice'], kind = 'reg')
sns.jointplot(train_data['LotArea'], train_data['SalePrice'], kind = 'reg')


# OpenPorch also seems to have 1 outlier. Lets clean 1 outlier from both LotFrontage and OpenPorchSF.

# In[ ]:


train_data = train_data.drop(train_data[(train_data['LotFrontage']>300) & (train_data['SalePrice']<400000)].index).reset_index(drop=True)
train_data = train_data.drop(train_data[(train_data['OpenPorchSF']>500) & (train_data['SalePrice']<100000)].index).reset_index(drop=True)
sns.jointplot(train_data['LotFrontage'], train_data['SalePrice'], kind = 'reg')
sns.jointplot(train_data['OpenPorchSF'], train_data['SalePrice'], kind = 'reg')


# So till now, I think  I have removed all the outliers from the Dataset. You can try for more if you think there are columns left.
# Note: I have clean outliers from top 20 numerical columns from which the outliers could be remove.

# So now let me check the plot of our target whether it is good or need to be medicated.

# In[ ]:


plt.figure(figsize = (9,6))
sns.distplot(train_data['SalePrice'])
plt.title('Dsitribution of SalesPrice')
plt.show()


# It seems to be unhealthy but I have a solution for this and a very easy one. I can take its log and I think it will be good to go(but why log? because it seems to be right skewed).

# In[ ]:


train_data['SalePrice'] = np.log(train_data['SalePrice'])


# In[ ]:


plt.figure(figsize = (9,6))
sns.distplot(train_data['SalePrice'])
plt.title('Dsitribution of SalesPrice')
plt.show()


# It looks pretty goood now.

# 
# It is time to check for the null values in the dataset. For this I will combine both training and test set and then I will manipulate null or missing values in the complete dataset.

# Before that I will take away the target variable 'SalePrice'

# In[ ]:


target = train_data['SalePrice']
train_data.drop(['SalePrice'], axis = 1, inplace = True)


# In[ ]:


full_data = pd.concat([train_data, test_data], axis = 0)
print('Shape of full_data will be:',full_data.shape)


# Now I have combined both the dataset, lets check for null or missing values in the dataset and try to infer from these null values and then I will plan for imputation strategy for each column which has null values.

# In[ ]:


miss_val = full_data.isnull().sum()
miss_full_data = miss_val[miss_val > 0].sort_values(ascending = False)
print('Columns with missing values:',len(miss_full_data))
print(miss_full_data)


# These are the number of missing values in the dataset and their plot in the following cell.

# In[ ]:


plt.figure(figsize = (16,9))
sns.barplot(y = miss_full_data.index, x = miss_full_data)
plt.title('Number of missing values in columns of Full Dataset')
plt.show()


# Now I will check that how many Categorical columns have missing values.

# In[ ]:


miss_cat = full_data[cat_cols].isnull().sum()
miss_full_cat = miss_cat[miss_cat > 0].sort_values(ascending = False)
print('Categorical Columns with missing values:',len(miss_full_cat))
print(miss_full_cat)


# In[ ]:


plt.figure(figsize = (16,7))
sns.barplot(y = miss_full_cat.index, x = miss_full_cat)
plt.title('Number of missing values in Categorical columns of Full Dataset')
plt.show()


# Same check for Numerical columns.

# In[ ]:


miss_num = full_data[num_cols].isnull().sum()
miss_full_num = miss_num[miss_num > 0].sort_values(ascending = False)
print('Numerical Columns with missing values:',len(miss_full_num))
print(miss_full_num)


# In[ ]:


plt.figure(figsize = (16,6))
sns.barplot(y = miss_full_num.index, x = miss_full_num)
plt.title('Number of missing values in Numerical columns of Full Dataset')
plt.show()


# ## Imputation Work

# For the imputation of missing values in a dataframe, the knowledge of dataset is must. eg- 'PoolQC' contains the highest number of missing values that is because most of the houses do not have pools in their houses, so I will replace their null values with 'None' or null values in garage and basement columns also make sense as they are not present in every house.

# Lets start with the cat columns as they have the most number of missing columns. In this list, most of the columns are self explanatory and in the description of those columns as well it is written that they are null because they are not present in the house so I will fill null values of those columns as 'None'.

# In[ ]:


none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MasVnrType']


# In[ ]:


print('Number of Categorical columns in which are going to be filled with "None":',len(none_cols))
for i in none_cols:
    print()
    print(full_data[i].value_counts())


# Filling these above columns with 'None' according to the description.

# In[ ]:


for i in none_cols:
    full_data[i] = full_data[i].fillna('None')


# In[ ]:


full_data[none_cols].isnull().sum()


# So, I have treated 15 out of 23 categorical columns which has null values(with just 'None' lol). Now for the remaining ones I will read their desciption and try to infer what could be the possible missing value for that column (this can be done with the help of domain knowledge as well).

# In[ ]:


cat_min_none = list(set(miss_full_cat.index) - set(none_cols))


# Now lets check what these remaining columns have got.

# In[ ]:


print('Categorical columns remaining are:',len(cat_min_none))
for i in cat_min_none:
    print()
    print(full_data[i].value_counts())


# All columns other than 'Utilities', I will fill with most_frequent values. I don't think that 'Utilities' will make much impact on 'SalePrice' because of the reason that there is only 1 'NoSeWa' value in that column that too in train set and rest are 'AllPub', apart from these there are only 2 null values so if I fill those with most_frequent then the whole column except one value will be 'AllPub'. So, I think it will be better dropping it from the dataset.

# In[ ]:


for i in cat_min_none:
    if i == 'Utilities':
        full_data.drop([i], axis = 1, inplace = True)
    else:
        full_data[i] = full_data[i].fillna(full_data[i].mode()[0])


# cleaning 'Utilities' from other lists as well because I will use these lists in future.

# In[ ]:


cat_min_none.remove('Utilities')
cat_cols.remove('Utilities')


# In[ ]:


full_data[cat_min_none].isnull().sum()


# So, upto this point all the categorical columns are free of null values.

# Now lets take care of Numerical columns.

# In[ ]:


miss_full_num = list(miss_full_num.index)
print('Number of Missing Numerical Columns:',len(miss_full_num))
print(miss_full_num)


# Apart from 'LotFrontage', I am filling all other columns with 0. 'LotFrontage' depends on the street area so I will take help from the 'Neighbourhood' column to fill it up.

# In[ ]:


full_data["LotFrontage"] = full_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for i in miss_full_num:
    if i != 'LotFrontage':
        full_data[i] = full_data[i].fillna(0)


# In[ ]:


full_data[miss_full_num].isnull().sum()


# I have removed all the null values from num_cols.

# So our dataset is cleaned now as all the missing values are now filled with appropriate entries

# Lets verify this by checking all the columns of a complete dataset.

# In[ ]:


full_data.info()

So, I have the complete data without any missing value. Now I will do a little bit of feature engineering in the data.
As discussed above after the correlation map, let us take some features out and merge some of them.
# In[ ]:


# full_data['Year_BuiltAndRemod'] = full_data['YearBuilt'] + full_data['YearRemodAdd']
# full_data['Total_SF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']
# full_data['Total_sqr_footage'] = full_data['BsmtFinSF1'] + full_data['BsmtFinSF2'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']
# full_data['Total_Bath'] = full_data['FullBath'] + (0.5 * full_data['HalfBath']) + full_data['BsmtFullBath'] + (0.5 * full_data['BsmtHalfBath'])
# full_data['Total_porch_SF'] = full_data['OpenPorchSF'] + full_data['3SsnPorch'] + full_data['EnclosedPorch'] + full_data['ScreenPorch'] + full_data['WoodDeckSF']


# ## Encoding and Transformation

# In[ ]:


print('Number of Numerical Columns:',len(num_cols))
print('Number of Categorical Columns:',len(cat_cols))


# In[ ]:


full_data[num_cols].head()


# I will take out some columns which are numerical in nature but are actually categorical. These includes 'MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold'.

# In[ ]:


# num_cat = ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']
# for i in num_cat:
#     full_data[i] = full_data[i].apply(str)
#skewness.drop(['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold'], axis = 0, inplace = True)


# I am taking out pure num columns.

# In[ ]:


# num_only = list(set(num_cols)-set(num_cat))


# Lets Scale the numerical columns with MinMaxScaler.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler(feature_range = (-1,1))
full_data[num_cols] = mm_scaler.fit_transform(full_data[num_cols])


# In[ ]:


full_data[num_cols].head()


# Lets start by looking at numerical data in our dataset.

# In[ ]:


# from scipy.stats import skew
# skewed_feats = full_data[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# print(skewness.shape)
# skewness


# This seems quite interesting, maximum of numerical columns does not seems to be normally distributed. Lets try to make them normal with the help of box cox transformation.

# I will use 'boxcox1p' to transform all the numerical columns. I am using it because it internally checks the type of transformation function which will be good for a particular column to make it normal.

# In[ ]:


# from scipy.special import boxcox1p
# skewness = skewness[abs(skewness) > 0.75]
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

# skewed_features = skewness.index
# lam = 0.15
# for feat in skewed_features:
#     full_data[feat] = boxcox1p(full_data[feat], lam)


# For the columns in num_cat, I will use OrdinalEncoder. You can use LabelEncoder as well but it is designed to encode one column at a time. OrdinalEncoder can encode mutiple columns at a time.

# In[ ]:


# from sklearn.preprocessing import OrdinalEncoder
# oe = OrdinalEncoder()
# full_data[num_cat] = oe.fit_transform(full_data[num_cat])


# After dealing with Numerical data lets see Categorical data now.

# In[ ]:


full_data[cat_cols].head()


# I will make use of FeaturHasher. I can use oneHot encoding too but that will increase the shape of the data and make features sparse.

# In[ ]:


from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features = 3, input_type = 'string')
df_cat_cols = pd.DataFrame()

for i in cat_cols:
    df_cat_cols = pd.concat([pd.DataFrame(fh.fit_transform(full_data[i]).toarray()).add_prefix(i+'_'),df_cat_cols], axis = 1)

df_cat_cols.head()


# So from above cell I have converted the categorical column into 3 columns and that 3 columns will be used to signify any value uniquely.
# Next, I will concat this dataframe into with dataframe of num_cols.

# In[ ]:


full_data = pd.concat([full_data[num_cols].reset_index(drop = True), df_cat_cols], axis = 1)


# Now, as the preprocessing has been done, I will take back the training and testing data for modelling.

# In[ ]:


train = full_data[:train_data.shape[0]]
test = full_data[train_data.shape[0]:]


# In[ ]:


print('Shape of Training set:',train.shape)
print('Shape of Test set:',test.shape)


# So, I have got my Training and Testing sets back. Let me take a look on them now.

# In[ ]:


train.head()


# In[ ]:


test = test.reset_index(drop = True)
test.head()


# In[ ]:


for i in train.columns:
    if train[i].dtype == 'object':
        print(i)


# No Output, that means all columns are in good condition to move into the model.

# In[ ]:


from sklearn.feature_selection import SelectKBest, mutual_info_regression
from functools import partial

num_cat_index = list(np.argwhere(train.columns.isin(num_cat)).ravel())
score_func = partial(mutual_info_regression, discrete_features = num_cat_index)
skb = SelectKBest(score_func)
skb_features = skb.fit_transform(train[num_cols], target)
skb_cols = list(skb.get_support(indices = True))


# In[ ]:


train = pd.concat([train,pd.DataFrame(skb_features, columns = range(0,10)).add_prefix('SKB_')], axis = 1)
test = pd.concat([test,pd.DataFrame(test.iloc[:,skb_cols].values, columns = range(0,10)).add_prefix('SKB_')], axis = 1)
train.drop(num_cols, axis = 1, inplace = True)
test.drop(num_cols, axis = 1, inplace = True)

print('Revised Shape of train set',train.shape)
print('Revised Shape of test set',test.shape)


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Modelling

# Now, I am splitting the training data into training and validation set.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train, target, test_size = 0.2)


# ### Models
# * Lasso
# * Elastic
# * GBMRegressor  (time consuming)
# * LGBMRegressor
# * XGBRegressor
# * RFRegressor

# ## More Libraries

# In[ ]:


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error


# Defining function for error calculations

# In[ ]:


def rmsle_func(actual, pred):
    return np.sqrt(((np.log(pred + 1) - np.log(actual + 1))**2).mean())

rmsle = make_scorer(rmsle_func, greater_is_better = False)


# In the following cell, I am building a function which will do all the hardwork of getting the best parameters for our models. For this I am using RandomizedSearchCV becuase it is faster than GridSearchCV.

# In[ ]:


def generate_clf(clf, params, x, y):
    rs = RandomizedSearchCV(clf, params)
    rs_obj = rs.fit(x, y)
    best_rs = rs_obj.best_estimator_
    print('Best parameters:',rs_obj.best_params_)
    pred = rs.predict(X_valid)
    print('Training score:',rs.score(x, y))
    print('Validation score:',rs.score(X_valid, y_valid))
    print('Validation RMSLE error:',rmsle_func(pred, y_valid))
    kf = KFold(n_splits = 5, shuffle = True)
    return np.sqrt(-1 * cross_val_score(best_rs, x, y, cv = kf, scoring = rmsle)) , rs_obj.best_params_


# I have created the Hard-working function, now I will define my models and call the above function with each of my models to get the best parameters.

# In[ ]:


from sklearn.linear_model import Lasso

# las_clf = Pipeline([('scaler', RobustScaler()),
#                     ('clf', Lasso())])
# las_clf = Lasso()

# las_params = {'alpha':[1e-4, 1e-3, 1e-2, 0.1, 0.05]}
# las_score , las_best_params = generate_clf(las_clf, las_params, train, target)
# print('Lasso RMSLE on Training set:',las_score.mean())


# After getting the best parameters for the model, I will pass these paramters to the final model.

# In[ ]:


las_clf_final = Lasso(alpha = 0.01)
# las_clf_final = Lasso(**las_best_params)


# The stuff I have done in last 2 cells, I will do the same for all the other models as well in now following cells.

# In[ ]:


from sklearn.linear_model import ElasticNet

# elas_clf = Pipeline([('scaler', RobustScaler()),
#                     ('clf', ElasticNet())])
# elas_clf = ElasticNet()

# elas_params = {'alpha':[1e-4, 1e-3, 1e-2, 0.1, 0.05],
#               'l1_ratio':[0.2, 0.4, 0.5, 0.6, 0.8]}
# elas_score , elas_best_params = generate_clf(elas_clf, elas_params, train, target)
# print('ElasticNet RMSLE on Complete Train set:',elas_score.mean())


# In[ ]:


elas_clf_final = ElasticNet(alpha = 0.01)
# elas_clf_final = ElasticNet(**elas_best_params)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

# gbm_clf = GradientBoostingRegressor()

# gbm_params = {'n_estimators':[2000, 2500, 3000, 3500, 4000],
#              'min_samples_split':[5, 10, 20, 30],
#              'max_depth':[3, 5, 7, 9],
#              'max_features':[100, 150, 200],
#              'learning_rate':[0.02, 0.04, 0.05, 0.1],
#              'loss': ['huber']}

# gbm_score , gbm_best_params = generate_clf(gbm_clf, gbm_params, train, target)
# print('GBM RMSLE on Complete Train set:',gbm_score.mean())


# In[ ]:


gbm_clf_final = GradientBoostingRegressor(loss = 'huber', n_estimators = 3000)
# gbm_clf_final = GradientBoostingRegressor(**gbm_best_params)


# In[ ]:


from xgboost import XGBRegressor

# xgb_clf = XGBRegressor()

# xgb_params = {'n_etimators':[2000, 2500, 3000, 3500, 4000],
#              'gamma':[0.02, 0.04, 0.05, 1],
#              'max_depth':[3, 5, 7, 9],
#              'alpha':[0.02, 0.04, 0.05, 1],
#              'eta':[0.02, 0.04, 0.05, 0.1]}

# xgb_score , xgb_best_params = generate_clf(xgb_clf, xgb_params, train, target)
# print('XGB RMSLE on Complete Train set:',xgb_score.mean())


# In[ ]:


xgb_clf_final = XGBRegressor(n_estimators = 3000)
# xgb_clf_final = XGBRegressor(**xgb_best_params)


# In[ ]:


from lightgbm import LGBMRegressor

# lgbm_clf = LGBMRegressor()

# lgbm_params = {'application':['regression'],
#               'num_iterations':[500, 700, 1000, 1500],
#               'max_depth':[3, 5, 7, 9],
#               'min_data_in_leaf':[4, 5, 6, 7, 8],
#               'feature_fraction':[0.2, 0.3, 0.4],
#               'bagging_fraction':[0.6, 0.7, 0.8, 0.9],
#                'num_leaves':[5, 7, 10],
#               'learning_rate':[0.01, 0.1, 0.02, 0.05, 0.03]}

# lgbm_score ,lgbm_best_params = generate_clf(lgbm_clf, lgbm_params, X_train, y_train)
# print('LGBM RMSLE on Complete Train set:',lgbm_score.mean())


# In[ ]:


lgbm_clf_final = LGBMRegressor(application = 'regression', num_iterations = 1000, max_depth = 7, num_leaves = 70)
# lgbm_clf_final = LGBMRegressor(**lgbm_best_params)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# rf_clf = RandomForestRegressor()

# rf_params = {'n_estimators':[300, 500, 1000, 1500, 2000],
#             'max_features':['auto', 'sqrt'],
#             'max_depth':[3, 5, 7, 9],
#             'min_samples_leaf':[50, 55, 60]}

# rf_score , rf_best_params = generate_clf(rf_clf, rf_params, train, target)
# print('RF RMSLE on train set:',rf_score.mean())


# In[ ]:


rf_clf_final = RandomForestRegressor(n_estimators = 1500, min_samples_leaf = 50, max_depth = 5)
# rf_clf_final = RandomForestRegressor(**rf_best_params)


# I have defined all my final models upto this point.
# In the next cell, I will be making a StackingRegressor to stack these models.

# In[ ]:


from sklearn.ensemble import StackingRegressor

my_estimators = [#('lgbm', lgbm_clf_final),
                ('xgb', xgb_clf_final)]

streg_clf = StackingRegressor(estimators = my_estimators, final_estimator = las_clf_final)
streg_clf.fit(train, target)
streg_clf_pred = np.exp(streg_clf.predict(test))
print(streg_clf.score(X_train, y_train))
print(streg_clf.score(X_valid, y_valid))
print(streg_clf.score(train, target))


# In[ ]:


# from sklearn.feature_selection import RFE

# rfe = RFE(estimator = rf_clf_final, n_features_to_select = 30)
# rfe.fit(train, target)
# rfe_final_cols = train.columns[rfe.support_]
# streg_clf_rfe = StackingRegressor(estimators = my_estimators, final_estimator = las_clf_final)
# streg_clf_rfe.fit(train[rfe_final_cols], target)
# streg_clf_pred_rfe = np.exp(streg_clf_rfe.predict(test[rfe_final_cols]))
# print(streg_clf_rfe.score(X_train[rfe_final_cols], y_train))
# print(streg_clf_rfe.score(X_valid[rfe_final_cols], y_valid))
# print(streg_clf_rfe.score(train[rfe_final_cols], target))


# In[ ]:


# from boruta import BorutaPy

# bor = BorutaPy(rf_clf_final, n_estimators = 'auto')
# bor.fit(train.values, target.values.ravel())
# bor_final_cols = train.columns[bor.support_]
# streg_clf_bor = StackingRegressor(estimators = my_estimators, final_estimator = las_clf_final)
# streg_clf_bor.fit(train[bor_final_cols], target)
# streg_clf_pred_bor = np.exp(streg_clf_bor.predict(test[bor_final_cols]))
# print(streg_clf_bor.score(X_train[bor_final_cols], y_train))
# print(streg_clf_bor.score(X_valid[bor_final_cols], y_valid))
# print(streg_clf_bor.score(train[bor_final_cols], target))


# ## Submission

# In[ ]:


submission_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission_data['SalePrice'] = streg_clf_pred
submission_data.to_csv('output.csv', index = False)


# In[ ]:


submission_data.info()


# In[ ]:


submission_data.head()


# In[ ]:




