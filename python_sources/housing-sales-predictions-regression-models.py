#!/usr/bin/env python
# coding: utf-8

# <h1 align=center><font size = 4>Housing Sales Prediction</font></h1>
# <h1 align=center><font size = 5>Regression Modelling</font></h1>

# # Table of Contents
# * [Setup](#setup)
# * [Get the Data](#get_data)
# * [Take a Quick Look at the Data Structure](#data_structure)
# * [Prepare Data for Machine Learning](#preparation)
# * [Select and Train a Model](#selection)
# * [Fine-tune the model](#tuning)
# * [Make Predictions](#predictions)

# <a id="setup"></a>
# # Setup

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
sns.set(style="darkgrid")

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# <a id="get_data"></a>
# # Get the Data

# In[ ]:


def load_housing_data(filename, house_path):
    csv_path = os.path.join(house_path, filename)
    return pd.read_csv(csv_path)


# In[ ]:


train_data = load_housing_data('train.csv',"../input")
test_data = load_housing_data('test.csv','../input')


# <a id="data_structure"></a>
# # Take a Quick Look at the Data Structure

# In[ ]:


housing = train_data.copy()


# In[ ]:


housing.info()


# In[ ]:


housing.head()


# In[ ]:


housing.describe()


# ### Categorical Attributes

# In[ ]:


#box plot overallqual/saleprice
var = 'MSZoning'
data = pd.concat([housing['SalePrice'], housing[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[ ]:


#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([housing['SalePrice'], housing[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# ### Numerical Attributes

# In[ ]:


housing['SalePrice'].describe()


# In[ ]:


sns.distplot(housing[['SalePrice']].dropna())
plt.show()


# In[ ]:


#skewness and kurtosis
print("Skewness: %f" % housing['SalePrice'].skew())
print("Kurtosis: %f" % housing['SalePrice'].kurt())


# In[ ]:


housing[housing.isnull().any(axis=1)].head()


# In[ ]:


sns.pairplot(data=housing[['SalePrice','MasVnrArea','LotFrontage','BsmtFinSF1','BsmtFinSF2']].dropna())
plt.show()


# ### Correlation Matrix

# In[ ]:


corrmat = housing.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[ ]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(housing[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from statsmodels.stats.outliers_influence import variance_inflation_factor

class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 15 are "okay".
        # Above 15 is too high and so should be removed.
        self.thresh = thresh
        
        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        if impute:
            self.imputer = SimpleImputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        print(self.imputer)
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
        print(X.columns)
        return X


# In[ ]:


transformer = ReduceVIF(thresh=15)
num_attributes = list(transformer.fit_transform(housing.select_dtypes(include=['int64','float64'])).columns.values)


# In[ ]:


corrmat = housing[num_attributes].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# ### Missing Data

# In[ ]:


#missing data
total = housing.isnull().sum().sort_values(ascending=False)
percent = (housing.isnull().sum()/housing.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# The attributes with more than 15% missing data will be removed

# In[ ]:


missing_data[missing_data.Percent>=0.15]


# <a id="preparation"></a>
# # Prepare Data for Machine Learning

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical, categorical or datetime columns 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names]


# ### Handling Categorical Attributes

# In[ ]:


housing = train_data.copy()


# In[ ]:


housing.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1, inplace=True)


# In[ ]:


cat_attributes = housing.select_dtypes(include='object').columns
cat_attributes


# In[ ]:


from sklearn.pipeline import Pipeline

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(attribute_names=cat_attributes))
])


# In[ ]:


cat_pipeline.fit_transform(housing).info()


# In[ ]:


plt.figure(figsize=(15,3))

ax1=plt.subplot(131)
sns.countplot(x='BsmtQual', data=housing, ax=ax1)
plt.xlabel('BsmtQual')

ax2=plt.subplot(132)
sns.countplot(x='BsmtCond', data=housing, ax=ax2)
plt.xlabel('BsmtCond')

ax3=plt.subplot(133)
sns.countplot(x='BsmtFinType2', data=housing, ax=ax3)
plt.xlabel('BsmtFinType2')

plt.show()


# In[ ]:


class CategoricalImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[attr].value_counts().index[0] for attr in X], index=X.columns)
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# In[ ]:


from sklearn.pipeline import Pipeline

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(attribute_names=cat_attributes)),
    ('imputer', CategoricalImputer())
])


# In[ ]:


some_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
some_incomplete_rows[cat_attributes]


# In[ ]:


cat_pipeline.fit_transform(housing).loc[some_incomplete_rows.index]


# In[ ]:


cat_pipeline.fit_transform(housing).info()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(attribute_names=cat_attributes)),
    ('imputer', CategoricalImputer()),
    ("encoder", OneHotEncoder(sparse=False))
])


# In[ ]:


cat_pipeline.fit_transform(housing)


# In[ ]:


cat_pipeline.fit_transform(housing).shape


# ### Handling Numerical Attributes

# In[ ]:


housing = train_data.copy()


# In[ ]:


housing.select_dtypes(include=['int64','float64']).columns


# In[ ]:


housing.drop(['Id','SalePrice', 'LotFrontage'],axis=1, inplace=True)


# In[ ]:


transformer = ReduceVIF(thresh=15)
num_attributes = list(transformer.fit_transform(housing.select_dtypes(include=['int64','float64'])).columns.values)


# In[ ]:


num_attributes = housing.select_dtypes(include=['int64','float64']).columns.values
num_attributes


# In[ ]:


housing[num_attributes].info()


# In[ ]:


housing[num_attributes].hist(figsize=(20,12))
plt.show()


# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')


# In[ ]:


housing_transform = pd.DataFrame(imputer.fit_transform(housing[num_attributes]), columns=num_attributes)


# In[ ]:


housing_transform = housing_transform.apply(lambda x: np.sign(x) * np.log(1 + np.abs(x)))


# In[ ]:


housing_transform.hist(figsize=(20,12))
plt.show()


# In[ ]:


class LogModulusTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return np.apply_along_axis(lambda x: np.sign(x) * np.log(1 + np.abs(x)), 1, X)
        # return X.apply(lambda x: np.sign(x) * np.log(1 + np.abs(x)))


# In[ ]:


BsmtFullBath_ix = list(num_attributes).index('BsmtFullBath')
BsmtHalfBath_ix = list(num_attributes).index('BsmtHalfBath')
HalfBath_ix = list(num_attributes).index('HalfBath')

extra_attributes = ['TotalBath']

class CombinedNumAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        total_bath = X[:, BsmtFullBath_ix] + X[:, BsmtHalfBath_ix] + X[:, HalfBath_ix]
        return np.c_[X, total_bath]


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attributes)),
    ('imputer', SimpleImputer(strategy='mean')),
    ('attr_adder', CombinedNumAttributesAdder()),
    ('transformer', LogModulusTransformer()),
    ('scaler', StandardScaler()),
])


# In[ ]:


num_pipeline.fit_transform(housing)


# In[ ]:


num_pipeline.fit_transform(housing).shape


# ### Transformation Pipelines

# In[ ]:


from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
    ])


# In[ ]:


X_train = train_data.copy()


# In[ ]:


X_train['SalePrice'].hist()
plt.show()


# In[ ]:


X_train['SalePrice'].apply(lambda x: np.sign(x) * np.log(1 + np.abs(x))).hist()
plt.show()


# In[ ]:


X_train_prepared = preprocess_pipeline.fit_transform(X_train)
y_train = X_train['SalePrice'].apply(lambda x: np.log(x))


# In[ ]:


print(X_train_prepared.shape)
print(y_train.shape)


# In[ ]:


cat_encoder = cat_pipeline.named_steps['encoder']
cat_encoder.categories_[:10]


# In[ ]:


cat_one_hot_attribs =[] 
for cat in range(len(cat_encoder.categories_)):
    cat_one_hot_attribs = cat_one_hot_attribs + list(cat_encoder.categories_[cat])


# In[ ]:


attributes = list(num_attributes) + extra_attributes + cat_one_hot_attribs
attributes


# <a id="selection"></a>
# # Select and Train a Model

# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train_prepared, y_train)
y_pred = lin_reg.predict(X_train_prepared)
lin_mse = mean_squared_error(y_train, y_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:


from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(penalty=None, random_state=42, max_iter=1000, tol=1e-6)
sgd_reg.fit(X_train_prepared, y_train)
y_pred = sgd_reg.predict(X_train_prepared)
sgd_mse = mean_squared_error(y_train, y_pred)
sgd_rmse = np.sqrt(sgd_mse)
sgd_rmse


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train_prepared, y_train)
y_pred = tree_reg.predict(X_train_prepared)
tree_mse = mean_squared_error(y_train, y_pred)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42, n_estimators=100)
forest_reg.fit(X_train_prepared, y_train)
y_pred = forest_reg.predict(X_train_prepared)
forest_mse = mean_squared_error(y_train, y_pred)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[ ]:


from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(X_train_prepared, y_train)
y_pred = svm_reg.predict(X_train_prepared)
svm_mse = mean_squared_error(y_train, y_pred)
svm_rmse = np.sqrt(svm_mse)
svm_rmse


# <a id="tuning"></a>
# # Fine-tune the model

# In[ ]:


from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[ ]:


sgd_scores = cross_val_score(sgd_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=5)
sgd_rmse_scores = np.sqrt(-sgd_scores)
display_scores(sgd_rmse_scores)


# In[ ]:


tree_scores = cross_val_score(tree_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=5)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)


# In[ ]:


forest_scores = cross_val_score(forest_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=5)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[ ]:


svm_scores = cross_val_score(svm_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=5)
svm_rmse_scores = np.sqrt(-svm_scores)
display_scores(svm_rmse_scores)


# In[ ]:


plt.figure(figsize=(8, 4))
plt.plot([1]*5, np.sqrt(-tree_scores), ".")
plt.plot([2]*5, np.sqrt(-svm_scores), ".")
plt.plot([3]*5, np.sqrt(-forest_scores), ".")
plt.plot([4]*5, np.sqrt(-sgd_scores), ".")
plt.boxplot([np.sqrt(-tree_scores), np.sqrt(-svm_scores), np.sqrt(-forest_scores), np.sqrt(-sgd_scores)], labels=("Tree","SVM", 'Forest', 'SGD'))
plt.ylabel("Mean Squared Error", fontsize=14)
plt.show()


# ### Grid Search

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [30, 50, 100],
     'max_features': [30, 50, 100]},
    {'bootstrap': [False],
     'n_estimators': [30, 50, 100],
     'max_features': [30, 50, 100]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search_forest = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search_forest.fit(X_train_prepared, y_train)


# In[ ]:


grid_search_forest.best_params_


# In[ ]:


grid_search_forest.best_estimator_


# In[ ]:


cvres = grid_search_forest.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[ ]:


param_grid = [
        {'kernel': ['linear'], 'C': [0.1, 1.0, 10.]},
        {'kernel': ['rbf'], 'C': [0.1, 1.0, 10.],
         'gamma': [0.001, 0.01, 0.1]},
    ]

svm_reg = SVR()
grid_search_svm = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
grid_search_svm.fit(X_train_prepared, y_train)


# In[ ]:


grid_search_svm.best_params_


# In[ ]:


cvres = grid_search_svm.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[ ]:


grid_search_svm.best_estimator_


# ### Feature Importances

# In[ ]:


feature_importances = grid_search_forest.best_estimator_.feature_importances_


# In[ ]:


sorted(zip(feature_importances, attributes), reverse=True)


# In[ ]:


indices = np.argsort(feature_importances)[::-1]

ranking = pd.DataFrame({'features':[attributes[indices[f]] for f in range(len(attributes[:10]))],
                        'values': [feature_importances[indices[f]] for f in range(len(attributes[:10]))]})

ranking


# In[ ]:


plt.figure(figsize=(6, 6))
sns.barplot(x='values', y='features', data=ranking, orient='h', color='mediumseagreen')
plt.title('feature importances')
plt.show()


# <a id="predictions"></a>
# # Make Predictions

# In[ ]:


final_model = grid_search_svm.best_estimator_
final_model


# In[ ]:


X_test=test_data.copy()


# In[ ]:


X_test.info()


# In[ ]:


X_test_prepared = preprocess_pipeline.transform(X_test)


# In[ ]:


y_pred = final_model.predict(X_test_prepared)
print(y_pred.shape)
y_pred.ravel()


# In[ ]:


test_data['SalePrice'] = np.exp(y_pred.ravel())
test_data[['Id','SalePrice']].head()


# In[ ]:


test_data[['Id','SalePrice']].to_csv('submission.csv', index=False)

