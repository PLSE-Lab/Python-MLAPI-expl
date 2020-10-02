#!/usr/bin/env python
# coding: utf-8

# ## Download the data

# In[ ]:


import os
import tarfile
import urllib


# In[ ]:


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# In[ ]:


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[ ]:


fetch_housing_data()


# In[ ]:


import pandas as pd


# In[ ]:


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# ## Take a Quick Look at the Data Structure

# In[ ]:


housing = load_housing_data()
housing.head()


# In[ ]:


housing.info()


# In[ ]:


housing['ocean_proximity'].value_counts()


# In[ ]:


housing.describe()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# ## Create a Test Set

# In[ ]:


import numpy as np


# In[ ]:


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[ ]:


train_set, test_set = split_train_test(housing, 0.2)


# In[ ]:


len(train_set)


# In[ ]:


len(test_set)


# ### How work a split data

# In[ ]:


from zlib import crc32


# In[ ]:


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


# In[ ]:


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[ ]:


housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# ### Split data with Scikit-Learn

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# ### Histogram of income categories

# In[ ]:


housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
                               labels=[1, 2, 3, 4, 5])


# In[ ]:


housing['income_cat'].hist()


# ### Stratified sampling

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[ ]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


# In[ ]:


for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[ ]:


strat_test_set['income_cat'].value_counts() / len(strat_test_set)


# In[ ]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)


# ## Discover and Visualize the Data to Gain Insights

# In[ ]:


housing = strat_train_set.copy()


# ### Visualizing Geographical Data

# In[ ]:


housing.plot(kind='scatter', x='longitude', y='latitude')


# In[ ]:


housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)


# In[ ]:


housing.plot(kind='scatter', 
             x='longitude', 
             y='latitude', 
             alpha=0.4,
             s=housing['population']/100,
             label='population',
             figsize=(10,7),
             c='median_house_value',
             cmap=plt.get_cmap('jet'),
             colorbar=True)
plt.legend()


# ### Looking for Correlations

# In[ ]:


corr_matrix = housing.corr()


# In[ ]:


corr_matrix['median_house_value'].sort_values(ascending=False)


# ### Correlations with pandas

# In[ ]:


from pandas.plotting import scatter_matrix


# In[ ]:


attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[ ]:


housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)


# ### Experimenting with Attribute Combinations

# In[ ]:


housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']


# In[ ]:


housing.head()


# In[ ]:


corr_matrix = housing.corr()


# In[ ]:


corr_matrix['median_house_value'].sort_values(ascending=False)


# ## Prepare the Data for Machine Learning Algorithms

# In[ ]:


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows


# In[ ]:


sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1


# In[ ]:


sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2


# ### Data Cleaning

# In[ ]:


median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3


# In[ ]:


sample_incomplete_rows


# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")


# In[ ]:


housing_num = housing.drop("ocean_proximity", axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])


# In[ ]:


imputer.fit(housing_num)


# In[ ]:


imputer.statistics_


# In[ ]:


housing_num.median().values


# In[ ]:


X = imputer.transform(housing_num)


# In[ ]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)


# In[ ]:


housing_tr.loc[sample_incomplete_rows.index.values]


# In[ ]:


imputer.strategy


# In[ ]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)


# In[ ]:


housing_tr.head()


# In[ ]:


housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)


# In[ ]:


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


# In[ ]:


ordinal_encoder.categories_


# In[ ]:


from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[ ]:


housing_cat_1hot.toarray()


# In[ ]:


cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot


# In[ ]:


cat_encoder.categories_


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self  # nothing else to do
    
    def transform(self, X):
        
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# ### Transformation Pipelines

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[ ]:


num_pipeline =  Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])


# In[ ]:


housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[ ]:


from sklearn.compose import ColumnTransformer


# In[ ]:


num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']


# In[ ]:


full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])


# In[ ]:


housing_prepared = full_pipeline.fit_transform(housing)


# ## Select and Train a Model

# ### Training and Evaluating on the Training Set

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print('Predictions:', lin_reg.predict(some_data_prepared))
print('Labels', list(some_labels))


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# ### Better Evaluation Using Cross-Validation

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
tree_rmse_scores = np.sqrt(-scores)


# In[ ]:


def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())


# In[ ]:


display_scores(tree_rmse_scores)


# In[ ]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[ ]:


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# ## Fine-Tune Your Model

# ### Grid Search

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 5]}
]


# In[ ]:


forest_reg = RandomForestRegressor()


# In[ ]:


grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True,
                           n_jobs=-1)


# In[ ]:


grid_search.fit(housing_prepared, housing_labels)


# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.best_estimator_


# In[ ]:


cvres = grid_search.cv_results_


# In[ ]:


for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)


# ### Analyze the Best Models and Their Errors

# In[ ]:


feature_importances = grid_search.best_estimator_.feature_importances_


# In[ ]:


feature_importances


# In[ ]:


extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
cat_encoder = full_pipeline.named_transformers_['cat']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# ### Evaluate Your System on the Test Set

# In[ ]:


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[ ]:


from scipy import stats


# In[ ]:


confidence = 0.95


# In[ ]:


squared_errors = (final_predictions - y_test)**2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(), scale=stats.sem(squared_errors)))

