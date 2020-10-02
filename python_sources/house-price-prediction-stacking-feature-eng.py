#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction

# ![](https://vmcdn.ca/f/files/kamloopsmatters/images/stock-photos/housing/money-for-house.jpg;w=630)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from scipy.special import boxcox1p
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

import statsmodels.api as sm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/house-prices-advanced-regression-techniques"))

# Any results you write to the current directory are saved as output.


# # Read Required Files

# First, we read the train data, test data, sample submission, and data description.
# 
# Data description is printed to give us better view of the data.

# In[ ]:


train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
data_desc = open("../input/house-prices-advanced-regression-techniques/data_description.txt")
print(data_desc.read())


# In[ ]:


test_data.shape


# In[ ]:


sample_submission


# In[ ]:


train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


train_data.describe()


# # Recognize Missing Data

# As seen above, many features are not complete and there exists a lot of null values. We will handle the incomplete data in this section.

# In[ ]:


data_null = train_data.isnull().sum()/len(train_data) * 100
data_null = data_null.drop(data_null[data_null == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': data_null})
missing_data.head(10)


# In[ ]:


plt.subplots(figsize=(8,6))
plt.xticks(rotation='90')
sns.barplot(data_null.index, data_null)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Missing rate', fontsize=12)


# # Correlation

# There are so many feautures and we need to gain a better insight about what will be useful here.

# In[ ]:


plt.subplots(figsize=(8,8))
corr_data = train_data.corr()
sns.heatmap(corr_data, square=True)


# It looks like that the pair of 
# * GarageCars
# * GarageArea
# 
# and also the pair of 
# * TotalBsmtSF
# * 1stFlrSF
# 
# are highly correlated and one of them can be dropped.

# ### SalePrice Correlation Matrix

# In[ ]:


k = 10 # number of variables for heatmap
cols = corr_data.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
plt.subplots(figsize=(8,8))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# It looks like that SalePrice is highly correlated with OverallQual, GrLivArea, Garage Capacity, and Basement Size.
# There are also many other attributes affecting the SalePrice.

# In[ ]:


corr_data["SalePrice"].sort_values(ascending=False)


# Generally, 
# ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt", "YearRemodAdd", "GarageYrBlt", "MasVnrArea", "Fireplaces"] 
# seem to be the most important factors.

# Relationship with "OverallQual" attribute:

# In[ ]:


var = 'OverallQual'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# SalePrice seems to increase with OverallQual.

# # Handling Missing Data

# Let's have a closer look at missing data.

# In[ ]:


missing_data


# I consider deleting the columns with more than 20% of missing data as thay are highly incomplete. 
# So we will delete the columns below:
# * PoolQc 
# * MiscFeature
# * Alley
# * Fence

# The rest can be replaced by strategies such as median.

# Let's look at which attributes are categorical.

# In[ ]:


train_data.select_dtypes(include=['object']).columns


# In[ ]:


train_data.select_dtypes(exclude=['object']).columns


# # Delete Outliers

# Outliers can be easily misleading. We nee

# In[ ]:


y_train = train_data["SalePrice"]
data = train_data.drop(["SalePrice"], axis=1)


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = train_data['GrLivArea'], y = train_data['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# We can see that there are a few obvious outliers which have a large area but low price. We can safely remove them.

# In[ ]:


train_data.sort_values(by = 'GrLivArea', ascending = False)[:2]


# In[ ]:


train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)
train_data = train_data.drop(train_data[train_data['Id'] == 524].index)


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(x = train_data['GrLivArea'], y = train_data['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[ ]:


def find_outliers(model, X, y, sigma=3):

    # predict y values using model
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    # if predicting fails, try fitting the model first
    except:
        model.fit(X,y)
        y_pred = pd.Series(model.predict(X), index=y.index)
        
    # calculate residuals between the model prediction and true y values
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # calculate z statistic, define outliers to be where |z|>sigma
    z = (resid - mean_resid)/std_resid    
    outliers = z[abs(z)>sigma].index
    
    return outliers


# ## Change the Sequential Categories

# Some of the categorical values such as Excellent, Good, etc. are values which can be sorted and have meaning in relation to each other. So, it is not wise to use 1-hot-encoding to handle them.

# In[ ]:


def changeSeqCat(data):
    cols_ExGd = ['ExterQual','ExterCond','BsmtQual','BsmtCond',
                 'HeatingQC','KitchenQual','FireplaceQu','GarageQual',
                'GarageCond','PoolQC']

    dict_ExGd = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0}

    for col in cols_ExGd:
        data[col].replace(dict_ExGd, inplace=True)  

    # Remaining columns
    data['BsmtExposure'].replace({'Gd':4,'Av':3,'Mn':2,'No':1,'None':0}, inplace=True)

    data['CentralAir'].replace({'Y':1,'N':0}, inplace=True)

    data['Functional'].replace({'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0}, inplace=True)

    data['GarageFinish'].replace({'Fin':3,'RFn':2,'Unf':1,'None':0}, inplace=True)

    data['LotShape'].replace({'Reg':3,'IR1':2,'IR2':1,'IR3':0}, inplace=True)

    data['Utilities'].replace({'AllPub':3,'NoSewr':2,'NoSeWa':1,'ELO':0}, inplace=True)

    data['LandSlope'].replace({'Gtl':2,'Mod':1,'Sev':0}, inplace=True)
    
    return data


# ### Set Missing Data to None-Existant

# If a value is missing it is probably becuase that attribute does not exist in that house. For instance if attributes regarding garage are missing, it is likely that the house does not have a garage. So, it is not wise to always fill null values with median, etc. strategies.

# In[ ]:


def setMissingData(data):
    # MSZoning NA in pred. filling with most popular values
    data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])

    # LotFrontage  NA in all. I suppose NA means 0
    data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mean())
    # Converting OverallCond to str
    data.OverallCond = data.OverallCond.astype(str)
    
    data['Fence'] = data['Fence'].fillna('NA')
    data['Alley'] = data['Alley'].fillna('NA')
    data['Functional'] = data['Functional'].fillna('Typ')

    # MasVnrType NA in all. filling with most popular values
    data['MasVnrType'] = data['MasVnrType'].fillna(data['MasVnrType'].mode()[0])

    # BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
    # NA in all. NA means No basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        data[col] = data[col].fillna('None')

    # TotalBsmtSF  NA in pred. I suppose NA means 0
    data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(0)

    # KitchenQual NA in pred. filling with most popular values
    data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])

    # FireplaceQu  NA in all. NA means No Fireplace
    data['FireplaceQu'] = data['FireplaceQu'].fillna('None')

    # GarageType, GarageFinish, GarageQual  NA in all. NA means No Garage
    for col in ('GarageType', 'GarageFinish', 'GarageQual'):
        data[col] = data[col].fillna('None')

    # GarageCars  NA in pred. I suppose NA means 0
    data['GarageCars'] = data['GarageCars'].fillna(0.0)

    # SaleType NA in pred. filling with most popular values
    data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])

    # Year and Month to categorical
    data['YrSold'] = data['YrSold'].astype(str)
    data['MoSold'] = data['MoSold'].astype(str)

    # Adding total sqfootage feature and removing Basement, 1st and 2nd floor features
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    
    data['YrBltAndRemod']=data['YearBuilt']+data['YearRemodAdd']

    data['Total_sqr_footage'] = (data['BsmtFinSF1'] + data['BsmtFinSF2'] +
                                     data['1stFlrSF'] + data['2ndFlrSF'])

    data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) +
                                   data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))

    data['Total_porch_sf'] = (data['OpenPorchSF'] + data['3SsnPorch'] +
                                  data['EnclosedPorch'] + data['ScreenPorch'] +
                                  data['WoodDeckSF'])
    
    data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    
    
    return data


# ## Handle Skewed Data

# In[ ]:


from scipy.special import boxcox1p
from scipy.stats import norm, skew #for some statistics

def handleSkew(all_data):
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness.head(10)
    skewness = skewness[abs(skewness) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        #all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)
    return all_data


# # Pipeline

# In[ ]:


class LabelBinarizerPipelineFriendly(MultiLabelBinarizer):
    def fit(self, X, y=None):
        super(LabelBinarizerPipelineFriendly,self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)
    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)


# In[ ]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attr):
        self.attributes = attr
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attributes].values


# In[ ]:


class LabelEncoderCat(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        X = X.apply(LabelEncoder().fit_transform)
        return X


# In[ ]:


id_test = test_data['Id']
data = setMissingData(data)
data = changeSeqCat(data)
data = handleSkew(data)


# In[ ]:


cat_attr = ['MSZoning', 'LandContour', 'Neighborhood', 'Condition1',
       'BldgType', 'RoofMatl', 'Exterior1st', 'LotConfig', 'Alley',
       'MasVnrType', 'BsmtFinType1', 'BsmtFinType2', 'Electrical',
       'GarageType', 'SaleCondition', 'BsmtFinType1', 'RoofStyle']

num_attr = ['LotFrontage', 'LotArea', 'OverallQual', 'TotalSF', 'CentralAir',
        'YearBuilt', 'BsmtFinSF1', 'BsmtUnfSF', 'BsmtExposure', 'GarageArea',
        'GrLivArea', 'BsmtFullBath', 'FullBath', 'TotRmsAbvGrd', 'GarageCond',
        'Fireplaces', 'WoodDeckSF','OpenPorchSF', 'GarageArea', 'GarageCars',
        'haspool', 'has2ndfloor', 'hasgarage', 'hasbsmt', 'hasfireplace', 
        'YrBltAndRemod', 'Total_sqr_footage', 'Total_Bathrooms', 'Total_porch_sf',
        'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'CentralAir',
        'BsmtCond', 'BsmtQual', 'OverallCond', 'Functional', 'ExterQual']
        

# cat_seq_attr = ['HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
#                 'BsmtCond', 'BsmtQual', 'OverallCond']

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attr)),
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
    ]) 

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attr)),
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('label_binarizer', LabelBinarizerPipelineFriendly()),
#         ('scaler', StandardScaler()),
    ])


full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])


# # RMSE Calculator

# We will now calculate the rmse using cross validation.

# In[ ]:


n_folds = 5

def rmse_cv(model, X_data, y_data):
    rmse= np.sqrt(-cross_val_score(model, X_data, y_data, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# # Prepare Data for Train

# In[ ]:


X_prepared = pd.DataFrame(full_pipeline.fit_transform(data))
X_prepared = pd.DataFrame(X_prepared)


# ### P-value

# The attributes with p-value less than 0.5 are useful. Note that if one of the classes of a categorical attribute has a desirable p-value, you should keep all the classes corresponding to that categorical attribute.

# In[ ]:


# from sklearn.feature_selection import f_regression

# all_cat = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
#        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
#        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
#        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
#        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
#        'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual',
#        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
#        'PavedDrive', 'Fence', 'SaleType', 'SaleCondition']

# not_evaluated = (list(set(all_cat) - set(cat_attr)))


# In[ ]:


data_null = data[cat_attr].isnull().sum()/len(data[cat_attr]) * 100
data_null = data_null.drop(data_null[data_null == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': data_null})
missing_data


# In[ ]:


from sklearn.feature_selection import f_regression

final_data = pd.get_dummies(data[cat_attr]).reset_index(drop=True)
F, p_value = f_regression(final_data, y_train)
np.array(final_data.columns) + " = " + (p_value < 0.05).astype(str)


# In[ ]:



outliers = find_outliers(Ridge(), X_prepared, y_train)
X_prepared = X_prepared.drop(outliers)
y_train = y_train.drop(outliers)

y_train = np.log1p(y_train)


# # Train Model

# * ### Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
score = rmse_cv(lin_reg, X_prepared, y_train)
score.mean()


# * ### Ridge Regression

# In[ ]:


ridge_reg = Ridge(alpha=0.001, solver="cholesky")
score = rmse_cv(ridge_reg, X_prepared, y_train)
score.mean()


# * ### Lasso Regression

# In[ ]:


lasso_reg = Lasso(alpha=0.0003)
score = rmse_cv(lasso_reg, X_prepared, y_train)
score.mean()


# * ### Elastic Net

# In[ ]:


elastic_net = ElasticNet(alpha=0.0005, l1_ratio=0.65)
score = rmse_cv(elastic_net, X_prepared, y_train)
score.mean()


# * ### RANSAC Regressor

# In[ ]:


ransac_reg = RANSACRegressor(lasso_reg)
score = rmse_cv(ridge_reg, X_prepared, y_train)
score.mean()


# * ### Random Forest

# In[ ]:


random_reg = RandomForestRegressor(max_depth=10, random_state=42, n_estimators=200)
score = rmse_cv(random_reg, X_prepared, y_train)
score.mean()


# * ### Huber Regressor

# In[ ]:


huber_reg = HuberRegressor(epsilon=5)
score = rmse_cv(huber_reg, X_prepared, y_train)
score.mean()


# * ### SVM

# In[ ]:


svm_reg = LinearSVR(epsilon=0.001)
score = rmse_cv(svm_reg, X_prepared, y_train)
score.mean()


# * ### Ada Boost

# In[ ]:


adf_clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=3000, learning_rate=0.05, 
                            loss='square')

score = rmse_cv(adf_clf, X_prepared, y_train)
score.mean()


# * ### Gradient Boosting

# In[ ]:


from sklearn.metrics import mean_squared_error

gbrt = GradientBoostingRegressor(max_depth=3, n_estimators=3000, learning_rate=0.05,
                                 max_features='sqrt', loss='huber')
gbrt.fit(X_prepared, y_train)
errors = [mean_squared_error(y_train, y_pred) for y_pred in gbrt.staged_predict(X_prepared)]
bst_n_estimators = np.argmin(errors)
bst_n_estimators


# In[ ]:


gbrt = GradientBoostingRegressor(max_depth=3, n_estimators=bst_n_estimators, learning_rate=0.05, max_features='sqrt',
                                 loss='huber')
score = rmse_cv(gbrt, X_prepared, y_train)
score.mean()


# * ### Voting Regressor

# In[ ]:


adf_clf.fit(X_prepared, y_train)
gbrt.fit(X_prepared, y_train)
lasso_reg.fit(X_prepared, y_train)
elastic_net.fit(X_prepared, y_train)
random_reg.fit(X_prepared, y_train)
ridge_reg.fit(X_prepared, y_train)
svm_reg.fit(X_prepared, y_train)
ransac_reg.fit(X_prepared, y_train)


# In[ ]:


from sklearn.ensemble import VotingRegressor

voting_reg = VotingRegressor([
            ("gradient_boost", gbrt),
            ("lasso_reg", lasso_reg),
            ("random_reg", random_reg),
#             ("ridge_reg", ridge_reg),
#             ("svm_reg", svm_reg),
            ("elastic_net", elastic_net),
            ]) 

score = rmse_cv(voting_reg, X_prepared, y_train)
score.mean()


# In[ ]:


voting_reg.fit(X_prepared, y_train)


# * ### Stacking

# In[ ]:


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    def fit(self, X, y=None):
        X = X.values
        y = y.values
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    
    def get_metafeatures(self, X):
        return np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
    
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# In[ ]:


stacked_averaged_models = StackingAveragedModels(base_models = (elastic_net, gbrt, ridge_reg, random_reg),
                                                 meta_model = lasso_reg)

score = rmse_cv(stacked_averaged_models, X_prepared, y_train)
score.mean()


# # Test

# In[ ]:


X_prepared.head()


# In[ ]:


test_data = setMissingData(test_data)
test_data = changeSeqCat(test_data)
test_data = handleSkew(test_data)


# In[ ]:


test_prepared = full_pipeline.transform(test_data)
test_prepared = pd.DataFrame(test_prepared)


# In[ ]:


stacked_averaged_models.fit(X_prepared, y_train)
stack_y_pred = stacked_averaged_models.predict(test_prepared)

voting_reg_y_pred = voting_reg.predict(test_prepared)


# In[ ]:


y_pred = np.expm1(stack_y_pred) * 0.75 + np.expm1(voting_reg_y_pred) * 0.25


# In[ ]:


submission = pd.DataFrame({
        "Id": id_test,
        "SalePrice": y_pred
    })

submission


# In[ ]:


submission.to_csv('prediction.csv', index=False)

