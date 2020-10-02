#!/usr/bin/env python
# coding: utf-8

# #### This is another Kaggle competition to test our skill of Regression techniques.
# 
# #### I am sharing my way of solving this problem :)
# 
# #### Import the libraries.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('figure', max_open_warning = 0)
import seaborn as sns
from pandas import DataFrame
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Read the datasets.

# In[ ]:


train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_data  = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# In[ ]:


train_data.head()


# #### Analyse the datasets.

# In[ ]:


train_data.shape


# In[ ]:


train_data.info()


# In[ ]:


test_data.shape


# In[ ]:


test_data.info()


# #### So we can see there are 81 columns with different data types and some of them contain missing values which we need to impute and the variable to predict is SalePrice.
# 
# #### Let's analyse the 'SalePrice' variable from train dataset.

# In[ ]:


fig = plt.figure(figsize=(10,5))
sns.distplot(train_data['SalePrice'])
plt.tight_layout()
plt.show()


# #### We can see the data is skewed to the right i .e positively skewed which is eveident from the Quantile-Quantile plot as displayed below.

# In[ ]:


from scipy import stats
stats.probplot(train_data['SalePrice'], plot=plt)


# #### Another important observation is 2 outliers at the top-right corner which we need to take care.

# #### To minimise the skewness we will apply log transformation.

# In[ ]:


fig = plt.figure(figsize=(10,5))
sns.distplot(np.log1p(train_data['SalePrice']))
plt.tight_layout()
plt.show()


# In[ ]:


stats.probplot(np.log1p(train_data['SalePrice']), plot=plt)


# #### From this visualizations we can infer the log transformation makes the data symmetrical.

# #### Since the datasets have mixed data we need to separate them out into multiple groups so that we can apply different imputation and transformation strategies.

# In[ ]:


cat_data = train_data.select_dtypes(include='object')
cat_cols = cat_data.columns

num_data = train_data.select_dtypes(exclude='object')
num_cols = num_data.columns

num_to_cat_cols = ['MSSubClass', 'MoSold', 'YrSold', 'OverallQual', 'OverallCond']

num_cols = [i for i in num_cols if not i in num_to_cat_cols]
num_cols = [i for i in num_cols if not i in ['Id']]

num_data = num_data.drop(['Id', 'MSSubClass', 'MoSold', 'YrSold', 'OverallQual', 'OverallCond'], axis=1)

print("There are %d Num , %d Cat, %d Num-Cat columns." % (len(num_cols), len(cat_cols), len(num_to_cat_cols)))


# #### Let's see the frequency of values of each categorical variable.

# In[ ]:


for i in range(len(cat_data.columns)):
    f, ax = plt.subplots(figsize=(7, 4))
    fig = sns.countplot(cat_data.iloc[:,i].dropna())
    plt.xlabel(cat_data.columns[i])
    plt.xticks(rotation=60)


# #### Since in each categorical variable the value of majority class is high compared to other class we will impute the missing values with most frequent class in each variablebut for some cases we will impute them as per the default value in field description.

# #### Univariate analysis of numeric columns which will give us a hint about their distribution.

# In[ ]:


for i in range(len(num_data.columns)):
    f, ax = plt.subplots(figsize=(7, 4))
    fig = sns.distplot(num_data.iloc[:,i].dropna(), rug=False, hist=False, kde_kws={'bw':0.1})
    plt.xlabel(num_data.columns[i])


# In[ ]:


skew_dict = {}
for cols in num_cols:
    skew_dict[cols] = {'Skewness': train_data[cols].skew()}
    
skew_df = pd.DataFrame(skew_dict).transpose()
skew_df.columns = ['Skewness']
skew_df.sort_values(by=['Skewness'], ascending=False)


# #### We can see the distribution is not symmetric so we need to transform the data to smooth.

# #### Let's see the correlation of numerical columns with SalePrice (Bi-variate Analysis).

# In[ ]:


fig = plt.figure(figsize=(12,18))
for i in range(len(num_data.columns)):
    fig.add_subplot(9, 4, i+1)
    sns.scatterplot(num_data.iloc[:, i], num_data['SalePrice'])
plt.tight_layout()
plt.show()


# #### We can see the co-relation co-efficients as well.

# In[ ]:


corr_matrix = train_data[num_cols].corr()
corr_matrix['SalePrice'].sort_values(ascending=False)


# In[ ]:


sns.heatmap(corr_matrix)


# #### Outlier identification.

# In[ ]:


sns.lmplot('GrLivArea', 'SalePrice', data=train_data, height=8)
plt.title("GrLivArea vs SalePrice")
plt.show()


# #### So there are 2 rows where GrLivArea is more than 4000 but prices are less than 200,000.
# #### Let's see their Id.

# In[ ]:


train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 300000)]


# In[ ]:


train_data = train_data.drop(train_data[train_data['Id'] == 524].index)
train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)


# #### Missing value imputation.

# In[ ]:


from sklearn.impute import SimpleImputer

mean_imputer    = SimpleImputer(strategy='mean')
median_imputer = SimpleImputer(strategy='median')
freq_imputer_MSZ    = SimpleImputer(strategy='most_frequent')
freq_imputer_UTI    = SimpleImputer(strategy='most_frequent')
freq_imputer_EXT1    = SimpleImputer(strategy='most_frequent')
freq_imputer_EXT2    = SimpleImputer(strategy='most_frequent')
freq_imputer_ELE    = SimpleImputer(strategy='most_frequent')
freq_imputer_KIT    = SimpleImputer(strategy='most_frequent')
freq_imputer_FUN    = SimpleImputer(strategy='most_frequent')
freq_imputer_SAL    = SimpleImputer(strategy='most_frequent')

train_data['MSZoning'] = freq_imputer_MSZ.fit_transform(train_data[['MSZoning']])
train_data['Alley']    = train_data['Alley'].fillna('No alley access')
train_data['Utilities']    = freq_imputer_UTI.fit_transform(train_data[['Utilities']])
train_data['Exterior1st']    = freq_imputer_EXT1.fit_transform(train_data[['Exterior1st']])
train_data['Exterior2nd']    = freq_imputer_EXT2.fit_transform(train_data[['Exterior2nd']])
train_data['MasVnrType']    = train_data['MasVnrType'].fillna('None')
train_data['BsmtQual']    = train_data['BsmtQual'].fillna('No Basement')
train_data['BsmtCond']    = train_data['BsmtCond'].fillna('No Basement')
train_data['BsmtExposure']    = train_data['BsmtExposure'].fillna('No Basement')
train_data['BsmtFinType1']    = train_data['BsmtFinType1'].fillna('No Basement')
train_data['BsmtFinType2']    = train_data['BsmtFinType2'].fillna('No Basement')
train_data['Electrical']    = freq_imputer_ELE.fit_transform(train_data[['Electrical']])
train_data['KitchenQual']    = freq_imputer_KIT.fit_transform(train_data[['KitchenQual']])
train_data['Functional']    = freq_imputer_FUN.fit_transform(train_data[['Functional']])
train_data['FireplaceQu']    = train_data['FireplaceQu'].fillna('No Fireplace')
train_data['GarageType']    = train_data['GarageType'].fillna('No Garage')
train_data['GarageYrBlt']    = train_data['GarageYrBlt'].fillna(train_data['YearBuilt'])
train_data['GarageFinish']    = train_data['GarageFinish'].fillna('No Garage')
train_data['GarageQual']    = train_data['GarageQual'].fillna('No Garage')
train_data['GarageCond']    = train_data['GarageCond'].fillna('No Garage')
train_data['PoolQC']    = train_data['PoolQC'].fillna('No Pool')
train_data['Fence']    = train_data['Fence'].fillna('No Fence')
train_data['MiscFeature']    = train_data['MiscFeature'].fillna('None')
train_data['SaleType']    = freq_imputer_SAL.fit_transform(train_data[['SaleType']])

train_data['LotFrontage']    = median_imputer.fit_transform(train_data[['LotFrontage']])
train_data['MasVnrArea']    = train_data['MasVnrArea'].fillna(0)
train_data['BsmtFinSF1']    = train_data['BsmtFinSF1'].fillna(0)
train_data['BsmtFinSF2']    = train_data['BsmtFinSF2'].fillna(0)
train_data['BsmtUnfSF']    = train_data['BsmtUnfSF'].fillna(0)
train_data['TotalBsmtSF']    = train_data['TotalBsmtSF'].fillna(0)
train_data['BsmtFullBath']    = train_data['BsmtFullBath'].fillna(0)
train_data['BsmtHalfBath']    = train_data['BsmtHalfBath'].fillna(0)
train_data['GarageCars']    = train_data['GarageCars'].fillna(0)
train_data['GarageArea']    = train_data['GarageArea'].fillna(0)

train_data[num_to_cat_cols] = train_data[num_to_cat_cols].astype(str)


# In[ ]:


#Missing value imputation (test_data)
test_data['MSZoning'] = freq_imputer_MSZ.transform(test_data[['MSZoning']])
test_data['Alley']    = test_data['Alley'].fillna('No alley access')
test_data['Utilities']    = freq_imputer_UTI.transform(test_data[['Utilities']])
test_data['Exterior1st']    = freq_imputer_EXT1.transform(test_data[['Exterior1st']])
test_data['Exterior2nd']    = freq_imputer_EXT2.transform(test_data[['Exterior2nd']])
test_data['MasVnrType']    = test_data['MasVnrType'].fillna('None')
test_data['BsmtQual']    = test_data['BsmtQual'].fillna('No Basement')
test_data['BsmtCond']    = test_data['BsmtCond'].fillna('No Basement')
test_data['BsmtExposure']    = test_data['BsmtExposure'].fillna('No Basement')
test_data['BsmtFinType1']    = test_data['BsmtFinType1'].fillna('No Basement')
test_data['BsmtFinType2']    = test_data['BsmtFinType2'].fillna('No Basement')
test_data['Electrical']    = freq_imputer_ELE.transform(test_data[['Electrical']])
test_data['KitchenQual']    = freq_imputer_KIT.transform(test_data[['KitchenQual']])
test_data['Functional']    = freq_imputer_FUN.transform(test_data[['Functional']])
test_data['FireplaceQu']    = test_data['FireplaceQu'].fillna('No Fireplace')
test_data['GarageType']    = test_data['GarageType'].fillna('No Garage')
test_data['GarageYrBlt']    = test_data['GarageYrBlt'].fillna(test_data['YearBuilt'])
test_data['GarageFinish']    = test_data['GarageFinish'].fillna('No Garage')
test_data['GarageQual']    = test_data['GarageQual'].fillna('No Garage')
test_data['GarageCond']    = test_data['GarageCond'].fillna('No Garage')
test_data['PoolQC']    = test_data['PoolQC'].fillna('No Pool')
test_data['Fence']    = test_data['Fence'].fillna('No Fence')
test_data['MiscFeature']    = test_data['MiscFeature'].fillna('None')
test_data['SaleType']    = freq_imputer_SAL.transform(test_data[['SaleType']])

test_data['LotFrontage']    = median_imputer.transform(test_data[['LotFrontage']])
test_data['MasVnrArea']    = test_data['MasVnrArea'].fillna(0)
test_data['BsmtFinSF1']    = test_data['BsmtFinSF1'].fillna(0)
test_data['BsmtFinSF2']    = test_data['BsmtFinSF2'].fillna(0)
test_data['BsmtUnfSF']    = test_data['BsmtUnfSF'].fillna(0)
test_data['TotalBsmtSF']    = test_data['TotalBsmtSF'].fillna(0)
test_data['BsmtFullBath']    = test_data['BsmtFullBath'].fillna(0)
test_data['BsmtHalfBath']    = test_data['BsmtHalfBath'].fillna(0)
test_data['GarageCars']    = test_data['GarageCars'].fillna(0)
test_data['GarageArea']    = test_data['GarageArea'].fillna(0)

test_data[num_to_cat_cols] = test_data[num_to_cat_cols].astype(str)


# #### Feature engineering - we will create:
# 
# 1. Total Bathroom
# 2. Total SF
# 3. Total Porch

# In[ ]:


train_data['Total_Bathroom'] = train_data['BsmtFullBath'] + 0.5 * train_data['BsmtHalfBath'] +                                train_data['FullBath'] + 0.5 * train_data['HalfBath']
    
train_data['Total_SF'] = train_data['TotalBsmtSF'] + train_data['1stFlrSF'] +                          train_data['2ndFlrSF'] + train_data['GrLivArea']
    
train_data['Total_Porch'] = train_data['OpenPorchSF'] + train_data['EnclosedPorch'] +                             train_data['3SsnPorch'] + train_data['ScreenPorch']


# In[ ]:


test_data['SalePrice'] = 0
test_data['Total_Bathroom'] = test_data['BsmtFullBath'] + 0.5 * test_data['BsmtHalfBath'] +                               test_data['FullBath'] + 0.5 * test_data['HalfBath']
    
test_data['Total_SF'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] +                         test_data['2ndFlrSF'] + test_data['GrLivArea']
    
test_data['Total_Porch'] = test_data['OpenPorchSF'] + test_data['EnclosedPorch'] +                            test_data['3SsnPorch'] + test_data['ScreenPorch']


# In[ ]:


add_num_cols = ['Total_Bathroom', 'Total_SF', 'Total_Porch']
num_cols.extend(add_num_cols)


# #### For categorical variables we will create dummy variables so we need to merge train and test data to ensure that none of the category is left out; otherwise the model will fail.

# In[ ]:


model_data = pd.concat([train_data, test_data], axis=0, sort=None, ignore_index=True)


# In[ ]:


model_data.tail()


# #### Create Age of the house variable.

# In[ ]:


model_data['Age'] = model_data['YearRemodAdd'] - model_data['YearBuilt']
num_cols.extend(['Age'])


# In[ ]:


sns.lmplot('Age', 'SalePrice', data=model_data, height=8)
plt.title("Age vs SalePrice")
plt.show()


# #### Create some boolean variable.

# In[ ]:


model_data['HasPorch'] = model_data['Total_Porch'].apply(lambda x: 1 if x > 0 else 0)
model_data['HasGarage'] = model_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
model_data['HasPool'] = model_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
model_data['WasRemodeled'] = (model_data['YearRemodAdd'] != model_data['YearBuilt']).astype(np.int64)
model_data['IsNew'] = (model_data['YearBuilt'] > 2000).astype(np.int64)
model_data['IsComplete'] = (model_data['SaleCondition'] != 'Partial').astype(np.int64)


# In[ ]:


model_data.shape


# #### Apply LabelEncoder to categorical features.

# In[ ]:


from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(model_data[c].values)) 
    model_data[c] = lbl.transform(list(model_data[c].values))


# #### Smooth the numerical data - apply BoxCox transformation.

# In[ ]:


numeric_feats = model_data.dtypes[model_data.dtypes != "object"].index
numeric_feats = [elem for elem in numeric_feats if elem not in ('Id', 'SalePrice')]

from scipy.special import boxcox1p

lam = 0.15
for feat in numeric_feats:
    model_data[feat] = boxcox1p(model_data[feat], lam)


# In[ ]:


model_data.head()


# #### We are creating dummy variables; so we are removing 'Id' otherwise we will get dummy varibles created for 'Id' which we don't want.

# In[ ]:


model_data_new = model_data.copy()
model_data_new = model_data_new.drop(['Id'], axis=1)

model_data_new = pd.get_dummies(model_data_new)
model_data_updated = pd.concat([model_data['Id'], model_data_new], axis=1)

model_data_updated.head()


# #### Split train-test data as per their 'Id' in the original dataset.

# In[ ]:


model_train_data = model_data_updated[model_data_updated.Id < 1461]
model_test_data  = model_data_updated[model_data_updated.Id > 1460]
print("Train Data Shape: ", model_train_data.shape)
print("Test Data Shape : ", model_test_data.shape)


# In[ ]:


model_label_data = pd.DataFrame(model_train_data['SalePrice'])
model_label_data.columns=['SalePrice']
model_train_data = model_train_data.drop(['Id', 'SalePrice'], axis=1)
model_test_data  = model_test_data.drop(['Id', 'SalePrice'], axis=1)
print("Train Data Shape: ", model_train_data.shape)
print("Test Data Shape : ", model_test_data.shape)


# #### Scale numerical data.

# In[ ]:


from sklearn.preprocessing import RobustScaler

std_scaler = RobustScaler()

model_train_data[numeric_feats] = std_scaler.fit_transform(model_train_data[numeric_feats])
model_test_data[numeric_feats]  = std_scaler.transform(model_test_data[numeric_feats])


# #### Log transform 'SalePrice'.

# In[ ]:


model_label_data['SalePrice'] = np.log(model_label_data['SalePrice'])


# #### So we have prepared the data for the model and we will start applying it.

# #### LASSO.

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

lasso_model = Lasso(alpha=0.0005, max_iter=30000, random_state=42)
score = cross_val_score(lasso_model, model_train_data, model_label_data, scoring='neg_mean_squared_error', cv=10)
scores_lasso = np.sqrt(-score).mean()
print("LASSO RMSE: ", scores_lasso)
print("LASSO STD : ", np.sqrt(-score).std())


# In[ ]:


lasso_model.fit(model_train_data, model_label_data)


# #### Ridge.

# In[ ]:


from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=0.0005, max_iter=30000, random_state=42)
score = cross_val_score(ridge_model, model_train_data, model_label_data, scoring='neg_mean_squared_error', cv=10)
scores_ridge = np.sqrt(-score).mean()
print("Ridge RMSE: ", scores_ridge)
print("Ridge STD : ", np.sqrt(-score).std())


# In[ ]:


ridge_model.fit(model_train_data, model_label_data)


# #### ElasticNet.

# In[ ]:


from sklearn.linear_model import ElasticNet

enet_model = ElasticNet(alpha=0.0001, max_iter=30000, l1_ratio=0.6, random_state=42)
score = cross_val_score(enet_model, model_train_data, model_label_data, scoring='neg_mean_squared_error', cv=10)
scores_enet = np.sqrt(-score).mean()
print("ElasticNet RMSE: ", scores_enet)
print("ElasticNet STD : ", np.sqrt(-score).std())


# In[ ]:


enet_model.fit(model_train_data, model_label_data)


# #### RandomForest Regressor.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
score = cross_val_score(rf_model, model_train_data, model_label_data.values.ravel(), scoring='neg_mean_squared_error', cv=10)
scores_rf = np.sqrt(-score).mean()
print("RandomForestRegressor RMSE: ", scores_rf)
print("RandomForestRegressor STD : ", np.sqrt(-score).std())


# In[ ]:


rf_model.fit(model_train_data, model_label_data.values.ravel())


# #### GradientBoosting Regressor.
# #### Boosting is an ensemble method that can combine several weak learners into a strong learner. The idea is to train predictors sequentially, eah trying to correct its predecessor. Most common types of Boosting algorithms are -
# 
# 1. *Adaptive Boosting*: First base predictor is trained to make predictions on the training set. The relative weight of is then increased based on the model performance metrics. A second predictor is trained using updated weights and it makes predictions on training data.The weights are updated again and this process continues.
# 
# 2. *Gradient Boosting*: Gradient Bossting works in the same sequential way of Adaptive Boosting but instead of tweaking the instance weights at every iteration it tries to fit new predictor to the residual errors of the previous predictor.

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(learning_rate=0.05, n_estimators=1000,max_depth=3, random_state=42)
score = cross_val_score(gb_model, model_train_data, model_label_data.values.ravel(), scoring='neg_mean_squared_error', cv=10)
scores_gb = np.sqrt(-score).mean()
print("GradientBoostingRegressor RMSE: ", scores_gb)
print("GradientBoostingRegressor STD : ", np.sqrt(-score).std())


# In[ ]:


gb_model.fit(model_train_data, model_label_data.values.ravel())


# In[ ]:


from xgboost import XGBRegressor

xgb_model = XGBRegressor(learning_rate=0.05, n_estimators=2500, max_depth=3, random_state=42)
score = cross_val_score(xgb_model, model_train_data, model_label_data.values.ravel(), scoring='neg_mean_squared_error', cv=10)
scores_xgb = np.sqrt(-score).mean()
print("XGBRegressor RMSE: ", scores_xgb)
print("XGBRegressor STD : ", np.sqrt(-score).std())


# In[ ]:


xgb_model.fit(model_train_data, model_label_data.values.ravel())


# #### Stacking - This is another ensemble method called *Stacked Generalization*. Each predictor performs a regression task on a new instance and then a meta learner/blender takes this predictions as inputs and makes the final prediction.

# In[ ]:


from sklearn.ensemble import StackingRegressor

estimators = [('Lasso', lasso_model),
              ('Ridge', ridge_model),
              ('Elastic Net', enet_model),
              ('Random Forest', rf_model),
              ('Gradient Boosting', gb_model),
              ('XGBoosting', xgb_model)]

stack_reg = StackingRegressor(estimators=estimators, final_estimator=xgb_model)
score = cross_val_score(stack_reg, model_train_data, model_label_data.values.ravel(), scoring='neg_mean_squared_error', cv=10)
scores_stack = np.sqrt(-score).mean()
print("Stacking Regressor RMSE: ", scores_stack)
print("Stacking Regressor STD : ", np.sqrt(-score).std())


# In[ ]:


stack_reg.fit(model_train_data, model_label_data.values.ravel())


# #### Visualise the performance of different models.

# In[ ]:


results = pd.DataFrame({
    'Model':['Lasso',
            'Ridge',
            'ElasticNet',
            'Random Forest',
            'Gradient Boosting',
            'XGBoost',
            'Stack'],
    'RMSE':[scores_lasso,
            scores_ridge,
            scores_enet,
            scores_rf,
            scores_gb,
            scores_xgb,
            scores_stack
            ]})

sorted_result = results.sort_values(by='RMSE', ascending=True).reset_index(drop=True)
sorted_result


# In[ ]:


f, ax = plt.subplots(figsize=(14,8))
plt.xticks(rotation='45')
sns.barplot(x=sorted_result['Model'], y=sorted_result['RMSE'])
plt.xlabel('Model', fontsize=15)
plt.ylabel('RMSE', fontsize=15)
plt.ylim(0.10, 0.14)
plt.title('Model Performance', fontsize=15)


# #### Prediction

# In[ ]:


test_data_id = test_data['Id']
lasso_predict = np.exp(lasso_model.predict(model_test_data))
ridge_predict = np.exp(ridge_model.predict(model_test_data))
enet_predict  = np.exp(enet_model.predict(model_test_data))
rf_predict    = np.exp(rf_model.predict(model_test_data))
gb_predict    = np.exp(gb_model.predict(model_test_data))
xgb_predict   = np.exp(xgb_model.predict(model_test_data))
stack_predict = np.exp(stack_reg.predict(model_test_data))

mixed_predict = np.exp((0.30 * np.log(lasso_predict)) + (0.30 * np.log(enet_predict)) + (0.05 * np.log(xgb_predict)) + (0.05 * np.log(gb_predict)) + (0.20 * np.log(stack_predict)) + (0.05 * np.log(ridge_predict.ravel())) + (0.05 * np.log(rf_predict)))


# In[ ]:


res_lasso = DataFrame(lasso_predict)
res_lasso.columns = ['SalePrice']
result_lasso = pd.concat([test_data_id, res_lasso], axis=1)
result_lasso.to_csv("Submission_Lasso.csv", index=False)

res_ridge = DataFrame(ridge_predict)
res_ridge.columns = ['SalePrice']
result_ridge = pd.concat([test_data_id, res_ridge], axis=1)
result_ridge.to_csv("Submission_Ridge.csv", index=False)

res_enet = DataFrame(enet_predict)
res_enet.columns = ['SalePrice']
result_enet = pd.concat([test_data_id, res_enet], axis=1)
result_enet.to_csv("Submission_ElasticNet.csv", index=False)

res_rf = DataFrame(rf_predict)
res_rf.columns = ['SalePrice']
result_rf = pd.concat([test_data_id, res_rf], axis=1)
result_rf.to_csv("Submission_RandomForestRegressor.csv", index=False)

res_gb = DataFrame(gb_predict)
res_gb.columns = ['SalePrice']
result_gb = pd.concat([test_data_id, res_gb], axis=1)
result_gb.to_csv("Submission_GradientBoostingRegressor.csv", index=False)

res_xgb = DataFrame(xgb_predict)
res_xgb.columns = ['SalePrice']
result_xgb = pd.concat([test_data_id, res_xgb], axis=1)
result_xgb.to_csv("Submission_XGBoostRegressor.csv", index=False)

res_stack = DataFrame(stack_predict)
res_stack.columns = ['SalePrice']
result_stack = pd.concat([test_data_id, res_stack], axis=1)
result_stack.to_csv("Submission_StackingRegressor.csv", index=False)


# In[ ]:


res_mixed = DataFrame(mixed_predict)
res_mixed.columns = ['SalePrice']
result_mixed = pd.concat([test_data_id, res_mixed], axis=1)
result_mixed.to_csv("Submission_Mixed.csv", index=False)

