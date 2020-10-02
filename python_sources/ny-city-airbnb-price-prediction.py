#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
df_airbnb = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
print(df_airbnb.info())
df_airbnb['last_review'] = pd.to_datetime(df_airbnb['last_review'])
print(df_airbnb.sample(5))
df_airbnb.columns
df_airbnb['last_review_year'] = df_airbnb['last_review'].dt.year


# In[ ]:


print(df_airbnb.head(2).T)
df_airbnb.isna().sum()


# In[ ]:


# https://www.kaggle.com/saireddy12/explorartory-data-analysis-on-haberman-dataset
sns.FacetGrid(df_airbnb,col='neighbourhood_group').map(sns.kdeplot,'price').set(xscale = 'log').add_legend()
plt.show()


# In[ ]:


# https://www.kaggle.com/saireddy12/explorartory-data-analysis-on-haberman-dataset
sns.FacetGrid(df_airbnb,col='room_type').map(sns.kdeplot,'price').set(xscale = 'log').add_legend()
plt.show()


# In[ ]:


df_airbnb['host_id'].value_counts()[:5]


# In[ ]:


df_airbnb['host_id'].value_counts()[-5:]


# In[ ]:


print(df_airbnb.groupby('neighbourhood_group')['host_id'].nunique())
print(df_airbnb['price'].describe())
print((df_airbnb['price']==0).sum())
print(df_airbnb[(df_airbnb['price']==0)].index.values)
# https://stackoverflow.com/questions/52456874/drop-rows-on-multiple-conditions-in-pandas-dataframe
df_airbnb.drop(df_airbnb[(df_airbnb['price']==0)].index.values, inplace=True)


# In[ ]:


df_airbnb.info()


# In[ ]:


_=df_airbnb['price'].plot(kind='box', figsize=(4,4))


# In[ ]:


df_airbnb.groupby('neighbourhood_group')['price'].describe()


# In[ ]:


_=df_airbnb.groupby('neighbourhood_group')['price'].describe().plot(kind='bar', log=True, figsize=(12,5))


# In[ ]:


df_airbnb.groupby('neighbourhood_group')['availability_365'].describe()


# In[ ]:


_=df_airbnb.groupby('neighbourhood_group')['availability_365'].describe().plot(kind='bar', log=True, figsize=(12,5))


# In[ ]:


df_airbnb.groupby('neighbourhood_group')['number_of_reviews'].describe()


# In[ ]:


_=df_airbnb.groupby('neighbourhood_group')['number_of_reviews'].describe().plot(kind='bar', log=True, figsize=(12,5))


# In[ ]:


df_airbnb.groupby('neighbourhood_group')['minimum_nights'].describe()


# In[ ]:


df_airbnb.groupby('neighbourhood_group')['calculated_host_listings_count'].describe()


# In[ ]:


df_airbnb['neighbourhood'].nunique()


# In[ ]:


# https://www.kaggle.com/prazhant/predicting-wait-times-at-intersections
_=df_airbnb.groupby('neighbourhood_group')['neighbourhood'].nunique().plot(kind='barh', figsize=(8,5))


# In[ ]:


_=df_airbnb['room_type'].value_counts().plot(kind='barh', figsize=(8,5))


# In[ ]:


# https://datascience.stackexchange.com/questions/49553/combining-latitude-longitude-position-into-single-feature
from math import radians, cos, sin, asin, sqrt

def single_pt_haversine(lat, lng, degrees=True):
    """
    'Single-point' Haversine: Calculates the great circle distance
    between a point on Earth and the (0, 0) lat-long coordinate
    """
    r = 6371 # Earth's radius (km). Have r = 3956 if you want miles

    # Convert decimal degrees to radians
    if degrees:
        lat, lng = map(radians, [lat, lng])

    # 'Single-point' Haversine formula
    a = sin(lat/2)**2 + cos(lat) * sin(lng/2)**2
    d = 2 * r * asin(sqrt(a)) 

    return d


# In[ ]:


df_airbnb['harvesine_distance'] = [single_pt_haversine(latitude, longitude) for latitude, longitude in zip(df_airbnb['latitude'], df_airbnb['longitude'])]


# In[ ]:


df_airbnb['last_review_year'].value_counts()


# In[ ]:


df_airbnb['last_review_year'] = df_airbnb['last_review_year'].fillna(2011.0)


# In[ ]:


feature_cols = ['host_id', 'neighbourhood_group', 'harvesine_distance',
       'neighbourhood', 'latitude', 'longitude', 'room_type',
       'minimum_nights', 'number_of_reviews','last_review_year',
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365']
target_col = 'price'


# In[ ]:


df_airbnb[feature_cols].corr()


# In[ ]:


feature_cols_2 = ['host_id', 'neighbourhood_group',
       'neighbourhood', 'harvesine_distance', 'room_type',
       'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count',
       'availability_365', 'last_review_year']


# In[ ]:


# https://www.afternerd.com/blog/append-vs-extend/
_ = pd.plotting.scatter_matrix(df_airbnb[feature_cols_2+['price']],figsize=(22, 22))


# In[ ]:


neighbourhood_group_dummies = pd.get_dummies(df_airbnb['neighbourhood_group'], prefix="neighbourhood_group")


# In[ ]:


room_type_dummies = pd.get_dummies(df_airbnb['room_type'], prefix="room_type")


# In[ ]:


print(df_airbnb['neighbourhood'].value_counts()[:5])
neighbourhood = df_airbnb['neighbourhood'].value_counts()[:5].index.tolist()
print(neighbourhood)
# print(len(neighbourhood))


# In[ ]:


# df_airbnb['neighbourhood'].isin(neighbourhood)


# In[ ]:


df_airbnb['neighbourhood_replaced'] = df_airbnb['neighbourhood']


# In[ ]:


# https://stackoverflow.com/questions/25028944/pandas-dataframe-replace-every-value-by-1-except-0
df_airbnb.loc[~df_airbnb['neighbourhood_replaced'].isin(neighbourhood),'neighbourhood_replaced'] = 'other'


# In[ ]:


df_airbnb['neighbourhood_replaced'].value_counts()


# In[ ]:


neighbourhood_dummies = pd.get_dummies(df_airbnb['neighbourhood_replaced'], prefix="neighbourhood")


# In[ ]:


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
# 'host_id',
X = pd.concat([df_airbnb[['harvesine_distance', 'last_review_year',
       'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count',
       'availability_365']], room_type_dummies, neighbourhood_group_dummies, neighbourhood_dummies], axis=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = pd.DataFrame(sc.fit_transform(X), columns = X.columns)
sc.mean_


# In[ ]:


# https://stackoverflow.com/questions/38913965/make-the-size-of-a-heatmap-bigger-with-seaborn
fig, ax = plt.subplots(figsize=(10,10))
_ = sns.heatmap(X.corr(), xticklabels=X.columns, yticklabels=X.columns, ax=ax)


# In[ ]:


X.columns


# In[ ]:


X.info()


# In[ ]:


y = np.log1p(df_airbnb['price'])


# In[ ]:


# https://datatofish.com/statsmodels-linear-regression/
import statsmodels.api as sm
Xsm = sm.add_constant(X)
model = sm.OLS(y.values, Xsm).fit()
print(model.summary()) 


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=2)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_test, y_test))
print([(a,b) for a,b in zip(X_test.columns, reg.coef_)])
# print(reg.coef_)
r2_score(y_test, reg.predict(X_test), multioutput='variance_weighted') 


# In[ ]:


fig, axes = plt.subplots(figsize=(16,5))
_=axes.plot(y_test.index, y_test, marker='', color='olive', linewidth=2)
_=axes.plot(y_test.index, reg.predict(X_test), marker='', color='r', linewidth=2, linestyle='dashed', label="predicted")


# In[ ]:


from sklearn.linear_model import ElasticNetCV
ev = ElasticNetCV(cv=5, random_state=0)
ev.fit(X_train, y_train)
print(ev.alpha_)
r2_score(y_test, ev.predict(X_test), multioutput='variance_weighted') 


# In[ ]:


fig, axes = plt.subplots(figsize=(16,5))
_=axes.plot(y_test.index, y_test, marker='', color='olive', linewidth=2)
_=axes.plot(y_test.index, ev.predict(X_test), marker='', color='r', linewidth=2, linestyle='dashed', label="predicted")


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr =  RandomForestRegressor(max_depth=4, random_state=0,
                              n_estimators=500)
rfr.fit(X_train, y_train)
print([(a,b) for a,b in zip(X_test.columns, rfr.feature_importances_)])
r2_score(y_test, rfr.predict(X_test), multioutput='variance_weighted') 


# In[ ]:


X.columns


# In[ ]:


# https://marcotcr.github.io/lime/tutorials/Using%2Blime%2Bfor%2Bregression.html
# https://github.com/marcotcr/lime/issues/293
import lime
import lime.lime_tabular
categorical_features = [6,7,8,9,10,11,12,13,14,15,16,17,18,19]
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=y_train.values, categorical_features=categorical_features, verbose=True, mode='regression')
i = 0
exp = explainer.explain_instance(X_test.values[i], rfr.predict, num_features=5)


# In[ ]:


exp.show_in_notebook(show_table=True)


# In[ ]:


fig, axes = plt.subplots(figsize=(16,5))
_=axes.plot(y_test.index, y_test, marker='', color='olive', linewidth=2)
_=axes.plot(y_test.index, rfr.predict(X_test), marker='', color='r', linewidth=2, linestyle='dashed', label="predicted")


# In[ ]:


from lightgbm import LGBMRegressor
lgbm = LGBMRegressor(objective='regression', n_jobs=-1, random_state=0,
                     #  learning_rate = 0.1,
                       max_depth = 4,
                      # min_data_in_leaf = 5,
                      # alpha = 0.5
                       )
lgbm.fit(X_train, y_train, eval_set=(X_test,y_test), eval_metric="r2", early_stopping_rounds=10, verbose=True)
print([(a,b) for a,b in zip(X_test.columns, lgbm.feature_importances_)])
r2_score(y_test, lgbm.predict(X_test), multioutput='variance_weighted') 


# In[ ]:


fig, axes = plt.subplots(figsize=(16,5))
_=axes.plot(y_test.index, y_test, marker='', color='olive', linewidth=2)
_=axes.plot(y_test.index, lgbm.predict(X_test), marker='', color='r', linewidth=2, linestyle='dashed', label="predicted")


# In[ ]:


# Plot residuals
fig, axes = plt.subplots(figsize=(16,5))
_=axes.plot(y_test.index, y_test-lgbm.predict(X_test), marker='', color='olive', linewidth=2)


# In[ ]:


plt.figure(figsize=(8, 8))
_=plt.scatter(y_test, lgbm.predict(X_test))
_=plt.xlabel('y_test')
_=plt.ylabel('y_test_predicted')


# In[ ]:


g = sns.jointplot(x=y_test, y=lgbm.predict(X_test), kind="kde", xlim=(2, 8), ylim=(2, 8))
_ = g.set_axis_labels("Actual", "Predicted")


# In[ ]:


_ = sns.distplot(y_test-lgbm.predict(X_test))


# In[ ]:


# plt.hist(y_test, log=True, bins=40)


# In[ ]:


# plt.hist(lgbm.predict(X_test), log=True)


# In[ ]:


# # https://github.com/slundberg/shap
# import shap
# shap.initjs()
# explainer = shap.TreeExplainer(lgbm)
# shap_values = explainer.shap_values(X_test)

# # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])


# In[ ]:


# # visualize the training set predictions
# shap.force_plot(explainer.expected_value, shap_values, X_test)


# In[ ]:


# shap.summary_plot(shap_values, X_test)


# In[ ]:


categorical_features = [6,7,8,9,10,11,12,13,14,15,16,17,18,19]
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=y_train.values, categorical_features=categorical_features, verbose=True, mode='regression')
i = 0
exp = explainer.explain_instance(X_test.values[i], lgbm.predict, num_features=5)


# In[ ]:


exp.show_in_notebook(show_table=True)


# https://www.neuraxle.neuraxio.com/stable/examples/boston_housing_regression_with_model_stacking.html#sphx-glr-examples-boston-housing-regression-with-model-stacking-py

# In[ ]:


get_ipython().system('pip install neuraxle')


# In[ ]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA, FastICA, IncrementalPCA
from sklearn.ensemble import GradientBoostingRegressor


from sklearn.utils import shuffle

from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyShapePrinter
from neuraxle.steps.sklearn import SKLearnWrapper, RidgeModelStacking
from neuraxle.union import AddFeatures


# In[ ]:


# X = X.astype(np.float32)


# In[ ]:


# from sklearn.cross_decomposition import CCA
# from sklearn.svm import SVR
# from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFECV
# p = Pipeline([
#     NumpyShapePrinter(),
#     AddFeatures([
#         SKLearnWrapper(RFECV(SVR(kernel="linear"), step=1, cv=3)),
#        # SKLearnWrapper(SelectKBest(mutual_info_regression, k=3)),
#        # SKLearnWrapper(FastICA(n_components=3)),
#     ]),
#     NumpyShapePrinter(),
#     RidgeModelStacking([
#         SKLearnWrapper(GradientBoostingRegressor()),
#         SKLearnWrapper(GradientBoostingRegressor(n_estimators=500)),
#         SKLearnWrapper(GradientBoostingRegressor(max_depth=5)),
#         SKLearnWrapper(KMeans()),
#     ]),
#     NumpyShapePrinter(),
# ])

# print("Fitting on train:")
# p = p.fit(X_train, y_train)
# print("")

# print("Transforming train and test:")
# y_train_predicted = p.transform(X_train)
# y_test_predicted = p.transform(X_test)
# print("")

# print("Evaluating transformed train:")
# score = r2_score(y_train_predicted, y_train)
# print('R2 regression score:', score)
# print("")

# print("Evaluating transformed test:")
# score = r2_score(y_test_predicted, y_test)
# print('R2 regression score:', score)


# https://www.neuraxle.neuraxio.com/stable/examples/boston_housing_meta_optimization.html#sphx-glr-examples-boston-housing-meta-optimization-py

# In[ ]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from sklearn.utils import shuffle

from neuraxle.hyperparams.distributions import RandInt, LogUniform, Boolean
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.metaopt.random import RandomSearch
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyTranspose
from neuraxle.steps.sklearn import SKLearnWrapper
from neuraxle.union import AddFeatures, ModelStacking

p = Pipeline([
    AddFeatures([
        SKLearnWrapper(
            PCA(n_components=2),
            HyperparameterSpace({"n_components": RandInt(1, 4)})
        ),
        SKLearnWrapper(
            FastICA(n_components=2),
            HyperparameterSpace({"n_components": RandInt(1, 4)})
        ),
    ]),
    ModelStacking([
        SKLearnWrapper(
            GradientBoostingRegressor(),
            HyperparameterSpace({
                "n_estimators": RandInt(100, 500), "max_depth": RandInt(3, 7), "learning_rate": LogUniform(0.01, 1.0)
            })
        ),
        SKLearnWrapper(
            GradientBoostingRegressor(),
            HyperparameterSpace({
                "n_estimators": RandInt(100, 500), "max_depth": RandInt(3, 7), "learning_rate": LogUniform(0.01, 1.0)
            })
        ),
        SKLearnWrapper(
            GradientBoostingRegressor(),
            HyperparameterSpace({
                "n_estimators": RandInt(100, 500), "max_depth": RandInt(3, 7), "learning_rate": LogUniform(0.01, 1.0)
            })
        ),
        SKLearnWrapper(
            KMeans(),
            HyperparameterSpace({"n_clusters": RandInt(2, 5)})
        ),
    ],
        joiner=NumpyTranspose(),
        judge=SKLearnWrapper(
            Ridge(),
            HyperparameterSpace({"alpha": LogUniform(0.1, 10.0), "fit_intercept": Boolean()})
        ),
    )
])


# In[ ]:


def meta_fit(p):
    print("Meta-fitting on train:")
    p = p.meta_fit(X_train, y_train, metastep=RandomSearch(
        n_iter=2, higher_score_is_better=True
    ))
    print("")

    print("Transforming train and test:")
    y_train_predicted = p.transform(X_train)
    y_test_predicted = p.transform(X_test)
    print("")
    

    print("Evaluating transformed train:")
    score = r2_score(y_train_predicted, y_train)
    print('R2 regression score:', score)
    print("")

    print("Evaluating transformed test:")
    score = r2_score(y_test_predicted, y_test)
    print('R2 regression score:', score)
    return y_train_predicted, y_test_predicted
    
y_train_predicted, y_test_predicted = meta_fit(p)


# In[ ]:


p[1][0].get_hyperparams()


# In[ ]:


p[1][1].get_hyperparams()


# In[ ]:


p[1][2].get_hyperparams()


# In[ ]:


p[1][3].get_hyperparams()


# In[ ]:


fig, axes = plt.subplots(figsize=(16,5))
_=axes.plot(y_test.index, y_test, marker='', color='olive', linewidth=2)
_=axes.plot(y_test.index, y_test_predicted, marker='', color='r', linewidth=2, linestyle='dashed', label="predicted")


# In[ ]:


# Plot residuals
fig, axes = plt.subplots(figsize=(16,5))
_=axes.plot(y_test.index, y_test-y_test_predicted, marker='', color='olive', linewidth=2)


# In[ ]:


_ = sns.distplot(y_test-y_test_predicted)


# In[ ]:


# plt.hist(y_test, log=True, bins=20)


# In[ ]:


# plt.hist(y_test_predicted, log=True)


# In[ ]:


plt.figure(figsize=(8, 8))
_=plt.scatter(y_test, y_test_predicted)
_=plt.xlabel('y_test')
_=plt.ylabel('y_test_predicted')


# In[ ]:


g = sns.jointplot(x=y_test, y=y_test_predicted, kind="kde")
_ = g.set_axis_labels("Actual", "Predicted")


# In[ ]:


categorical_features = [6,7,8,9,10,11,12,13,14,15,16,17,18,19]
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=y_train.values, categorical_features=categorical_features, verbose=True, mode='regression')
i = 0
exp = explainer.explain_instance(X_test.values[i], p.predict, num_features=5)


# In[ ]:


exp.show_in_notebook(show_table=True)


# In[ ]:


exp.as_list()


# In[ ]:


# https://github.com/slundberg/shap
import xgboost
import shap

# load JS visualization code to notebook
shap.initjs()

# train XGBoost model

model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

