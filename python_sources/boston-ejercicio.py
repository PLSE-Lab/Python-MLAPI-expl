#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[ ]:


dataset = pd.read_csv("../input/california-housing-prices/housing.csv")


# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset['ocean_proximity'].value_counts()


# In[ ]:


dataset.describe()


# In[ ]:


#Lets plot whole dataset for a visual interpretation
get_ipython().run_line_magic('matplotlib', 'inline')
dataset.hist(bins=50, figsize=(20,18))


# In[ ]:


from sklearn.model_selection import train_test_split

train_set, test_set, = train_test_split(dataset, test_size = 0.2, random_state= 42)


# In[ ]:


dataset['income_cat'] = pd.cut(dataset['median_income'],
                                bins=[0.,1.5, 3.0, 4.5, 6., np.inf],
                                  labels=[1,2,3,4,5])


# In[ ]:


dataset['income_cat'].hist(figsize=(18,12))


# In[ ]:


#stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits =1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(dataset, dataset['income_cat']):
    strat_train_set = dataset.iloc[train_index]
    strat_test_set = dataset.iloc[test_index]
    


# In[ ]:


strat_test_set['income_cat'].value_counts() / len(strat_test_set)


# In[ ]:


for set_ in (strat_test_set, strat_train_set):
    set_.drop("income_cat", axis=1, inplace=True)


# In[ ]:


dataset = strat_train_set.copy()


# In[ ]:


dataset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(18,10))


# In[ ]:


dataset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s = dataset['population']/100, label="population", figsize=(14,8),
            c = "median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()


# In[ ]:


corr_matrix  = dataset.corr()


# In[ ]:


corr_matrix['median_house_value'].sort_values(ascending=False)


# In[ ]:


from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(dataset[attributes], figsize=(20,12))


# In[ ]:


dataset.plot(kind="scatter", x='median_income', y='median_house_value', alpha=0.2, figsize=(18,12))


# In[ ]:


dataset['rooms_per_household'] = dataset['total_rooms'] / dataset['households']
dataset['bedrooms_per_room'] = dataset['total_bedrooms'] / dataset['total_rooms']
dataset['population_per_household'] = dataset['population'] / dataset['households']


# In[ ]:


corr_matrix = dataset.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# In[ ]:


dataset = strat_train_set.drop("median_house_value", axis=1)
dataset_labels = strat_train_set['median_house_value'].copy()


# In[ ]:


from sklearn.impute import SimpleImputer


# In[ ]:


dataset_num = dataset.drop("ocean_proximity", axis=1)


# In[ ]:


imputer = SimpleImputer(strategy="median")
imputer.fit(dataset_num)


# In[ ]:


imputer.statistics_


# In[ ]:


dataset_num.median().values


# In[ ]:


X = imputer.transform(dataset_num)


# In[ ]:


X


# In[ ]:


#Categorical Variable to NUMERICAL value
from sklearn.preprocessing import OrdinalEncoder
dataset_cat = dataset[['ocean_proximity']]
ordinal_encoder = OrdinalEncoder()
dataset_encoded = ordinal_encoder.fit_transform(dataset_cat)


# In[ ]:


dataset_encoded.shape


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
dataset_cat_1hot  = cat_encoder.fit_transform(dataset_cat)
dataset_cat_1hot


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

bedrooms_ix, population_ix, rooms_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] /X[:,households_ix]
        population_per_houlsehold = X[:,population_ix]/ X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_houlsehold, 
                        bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household,population_per_houlsehold]
        
    


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
                            ('imputer', SimpleImputer(strategy='median')), 
                            ('attrib_adder',CombinedAttributesAdder()),
                            ('std_scaler', StandardScaler()) 
                        ])
                        


# In[ ]:


housing_num_tr = num_pipeline.fit_transform(dataset_num)


# In[ ]:


housing_num_tr


# In[ ]:


from sklearn.compose import ColumnTransformer
num_attribs = list(dataset_num)
cat_attribs = ['ocean_proximity']

full_pipeline= ColumnTransformer([("num", num_pipeline, num_attribs),
                                 ("cat", OneHotEncoder(), cat_attribs) ])


# In[ ]:


full_pipeline


# In[ ]:


housing_prepared = full_pipeline.fit_transform(dataset)


# In[ ]:


housing_prepared


# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, dataset_labels)


# In[ ]:


some_data = dataset.iloc[:5]
some_labels = dataset_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions: ", lin_reg.predict(some_data_prepared))


# In[ ]:


print("Labels",list(some_labels))


# In[ ]:


#calculate error

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(dataset_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, dataset_labels)


# In[ ]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(dataset_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[ ]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, dataset_labels,
                        scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[ ]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean() )
    print("Standard Deviation:", scores.std() )


# In[ ]:


display_scores(tree_rmse_scores)


# In[ ]:


lin_scores = cross_val_score(lin_reg, housing_prepared, dataset_labels,
                            scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)


# In[ ]:


display_scores(lin_rmse_scores)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor()
random_forest.fit(housing_prepared, dataset_labels)


# In[ ]:





# 

# In[ ]:


random_forest_scores = cross_val_score(random_forest, housing_prepared, dataset_labels,
                                      scoring = "neg_mean_squared_error", cv=10)
random_forest_scores_sqrt = np.sqrt(-random_forest_scores)
display_scores(random_forest_scores_sqrt)


# In[ ]:


from sklearn.externals import joblib

joblib.dump(random_forest, "random_forest_model.pkl")
joblib.dump(lin_reg, "linear_regression_model.pkl")
joblib.dump(tree_reg, "tree_regression_model.pkl")


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


param_grid = [
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]}
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring="neg_mean_squared_error",
                          return_train_score=True)
grid_search.fit(housing_prepared, dataset_labels)


param_distributions = {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]}

forest_reg = RandomForestRegressor()
cv_ = RandomizedSearchCV(forest_reg, param_distributions, cv=5, 
                        scoring = "neg_mean_squared_error",
                        return_train_score=True)
cv_.fit


# In[ ]:


cv_.score


# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.best_estimator_


# In[ ]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres['params']):
    print(np.sqrt(-mean_score), params)


# In[ ]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[ ]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_['cat']
cat_one_hot_encoder = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_encoder
sorted(zip(feature_importances, attributes),reverse=True)


# In[ ]:


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepraded = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepraded)

joblib.dump(final_model, 'final_model.pkl')


# In[ ]:


testi_ = pd.DataFrame({'longitude':[-142.28], 'latitude':[42.12], 'housing_median_age':[29],
                      'total_rooms':[432],'total_bedrooms':[233],'population':[2000],'households':[2432],'median_income':[3.421],
                      'ocean_proximity':['INLAND']})


# In[ ]:


testi_


# In[ ]:


prepared_testi = full_pipeline.transform(testi_)


# In[ ]:


load_model = joblib.load('random_forest_model.pkl')
load_model2 = joblib.load('linear_regression_model.pkl')
load_model3 = joblib.load('tree_regression_model.pkl')
load_model4 = joblib.load('final_model.pkl')

print(load_model.predict(prepared_testi),
        load_model2.predict(prepared_testi),
        load_model3.predict(prepared_testi),
        load_model4.predict(prepared_testi)
     )


# In[ ]:


final_model.predict(prepared_testi)


# In[ ]:


X_test


# In[ ]:


final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[ ]:


X_test_prepraded.shape


# In[ ]:


dataset['ocean_proximity'].value_counts()


# In[ ]:


from sklearn import svm

svm_reg = svm.SVR(kernel="linear")
svm_reg.fit(housing_prepared, dataset_labels)
predict_smv_reg = svm_reg.predict(X_test_prepraded)




# In[ ]:


predict_smv_reg
final_mse_svm = mean_squared_error(y_test, predict_smv_reg)
final_rmse_smv = np.sqrt(final_mse_svm)
final_rmse_smv


# In[ ]:


svm_reg = svm.SVR(kernel="rbf")
svm_reg.fit(housing_prepared, dataset_labels)
predict_smv_reg = svm_reg.predict(X_test_prepraded)


# In[ ]:


predict_smv_reg
final_mse_svm = mean_squared_error(y_test, predict_smv_reg)
final_rmse_smv = np.sqrt(final_mse_svm)
final_rmse_smv


# In[ ]:


#Necesario un attributo para la prediccion.


X_test_prepraded.shape


# In[ ]:


cv_.fit(housing_prepared, dataset_labels)


# In[ ]:


full_pipeline_with_predictor = Pipeline([("preparation", full_pipeline),
        ("linear", LinearRegression())])


# In[ ]:


full_pipeline_with_predictor.fit(dataset, dataset_labels)


# In[ ]:


full_pipeline_with_predictor.predict(some_data)


# In[ ]:





# In[ ]:




