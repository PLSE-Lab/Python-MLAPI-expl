#!/usr/bin/env python
# coding: utf-8

# # Getting the data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Reading the csv file using pandas read_csv() function and storing it in a housing variable

# In[ ]:


housing = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")


# Importing all the modules which are necessary

# In[ ]:


from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


housing.info()


# The ocean proximity contains categorical values so we are viewing the data using value_counts() to know the occurence of each value in the dataset.

# In[ ]:


housing['ocean_proximity'].value_counts()


# head() returns the first 5 values in the dataframe.

# In[ ]:


housing.head()


# # Data visualisation
# 

# Plotting a histogram to visualize and understand more about the data
# 

# In[ ]:


housing.hist(bins=50,figsize = (20,20))


# Since the correlation between the median_income is the important feature in predicting the house value we will ensure that while splitting the data into training and testing sets we will split them with the equal values of bins in the both the training and testing set.

# This can be done by splitting the median_income into different bins of values and then labelling it.

# In[ ]:


housing['income_cat'] = pd.cut(housing['median_income'],
                              bins = [0.,1.5,3.0,4.5,6.,np.inf],
                              labels = [1,2,3,4,5])


# Then using StratifiedShuffleSplit from sklearn.model_selection we will split the data into training and test sets based on the test_size.

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1 , test_size = 0.2 , random_state = 42)


# In the below code we will get the index for the train and test set splitting based on the income_cat column produced before.

# In[ ]:


for train_index , test_index in split.split(housing , housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# After splitting the data the income_cat column is dropped since there is no further use of it.

# In[ ]:


strat_train_set.drop("income_cat",axis = 1 ,inplace = True)
strat_test_set.drop("income_cat",axis = 1 ,inplace = True)


# In[ ]:


strat_train_set.columns


# In[ ]:


strat_test_set.columns


# Making a copy of the stratified training set to the housing_train

# In[ ]:


housing_train = strat_train_set.copy()


# Plotting the graph based on the latitude and longitude values.

# In[ ]:


housing_train.plot(kind = "scatter" , x = 'longitude' , y = 'latitude')


# In[ ]:


housing_train.plot(kind = "scatter" , x = 'longitude',y = 'latitude',alpha = 0.1)


# Here the size of the circle represents the size the population in that area and the color scale represents the median_house_value in that area with blue as the lowest and the red as the highest median_house_value.

# In[ ]:


housing_train.plot(kind = "scatter" , x = 'longitude',y = 'latitude',alpha = 0.4,
                  s = housing_train['population']/100 , label = 'population' , figsize = (10,7),
                  c = 'median_house_value' , cmap = plt.get_cmap("jet"),colorbar = True)
plt.legend()


# Finding the correlation between the different features.

# In[ ]:


corr_mat = housing_train.corr()
corr_mat


# Finding the correlation between the mediana_house_value with all other features.
# medain_income is high positively correlated with in median_house_value and the latitude is high negative correlated with the median_house_value.

# In[ ]:


corr_mat['median_house_value'].sort_values(ascending = False)


# Scatter matrix is an another type to view the correlation between different columns in the graph visualization.

# In[ ]:


from pandas.plotting import scatter_matrix

attr = ['median_house_value' , 'median_income' , 'total_rooms','housing_median_age']
scatter_matrix(housing_train[attr],figsize=(15,10))


# # Feature Engineering

# Creating new features based on the features that are already present.

# In[ ]:


housing_train['rooms_per_household'] = housing_train['total_rooms'] / housing_train['households']
housing_train['bedrooms_per_household'] = housing_train['total_bedrooms'] / housing_train['total_rooms']
housing_train['population_per_household'] = housing_train['population'] / housing_train['households']


# Now finding the correlation for the dataframe with new added columns.

# In[ ]:


corr_mat = housing_train.corr()


# Now the coorelation between the rooms_per_household and the median_house_value is high positive correlated and the bedrooms_per_household is high negative correlated which means that the feature created using the already existing ones make more sense.

# In[ ]:


corr_mat['median_house_value'].sort_values(ascending = False)


# Dropping the label value from the training set.

# In[ ]:


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# # Handling missing values

# Now we will get all the numerical columns from the dataframe and store it seperately.

# In[ ]:


housing_num = housing.drop('ocean_proximity',axis = 1)


# Using simpleImputer from the sklearn.impute we will fill all the numerical missing values with the median of that value.

# In[ ]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = 'median')

X = imputer.fit_transform(housing_num)


# In[ ]:


imputer.statistics_


# In[ ]:


X


# Converting the matrix which is returned from the imputer into the dataframe using the below code.

# In[ ]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing_num.index)
housing_tr


# In[ ]:


housing_train


# # Handling categorical variables

# Storing the categorical variable in the housing_cat.

# In[ ]:


housing_cat = pd.DataFrame(housing_train['ocean_proximity'])
housing_cat


# Using OneHotEncoder from the sklearn.preprocessing to encode all the categorical variables into numerical ones.

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()
housing_cat_1hot = onehotencoder.fit_transform(housing_cat)
housing_cat_1hot


# OneHotEncoder returns the result as a sparse matrix, we are converting that into an array using pandas toarray() method.

# In[ ]:


housing_cat_1hot.toarray()


# # Creating pipeline

# Creating a pipeline to preprocess the numerical data.

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('std_scaler', StandardScaler()),
 ])
housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[ ]:


from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
 ("num", num_pipeline, num_attribs),
 ("cat", OneHotEncoder(), cat_attribs),
 ])
housing_prepared = full_pipeline.fit_transform(housing)


# # Fitting the model

# Here we will create a Linear Regression model and fit with the training data.

# In[ ]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# predicting the value for the test sets in the linear regression model and using the error function root mean squared error.

# In[ ]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# As linear regression rmse is not very fine we will define a DecisionTreeRegressor and fit the model with training data and predict it.

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# The model has the rmse value of 0. which says that it will predict with 100% accuracy,but the model overfits the data so that the model only works well on the training data and predicts worst with the unknown values. 

# In[ ]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# To overcome the overfitting in the training set we will use cross_validation for the decision tree regressor. 

# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

tree_rmse_scores


# Now the rmse score is much worse than the linear regressor but its not overfitting the data.

# In[ ]:


tree_rmse_scores.mean()


# Using the cross validation for the linear regressor.

# In[ ]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
lin_rmse_scores


# In[ ]:


lin_rmse_scores.mean()


# Since both the models rmse values are too high we will try it with the RandomForestRegressor.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

forest_score = cross_val_score(forest_reg,housing_prepared,housing_labels,scoring = 'neg_mean_squared_error',cv=10)
forest_rmse_score = np.sqrt(-forest_score)
forest_rmse_score


# Hurray! RandomForestRegressor models rmse value is lower than the all other models but it also not the best score.

# In[ ]:


forest_rmse_score.mean()


# # GridSearchCV

# Using the GridSearchCV for finding the best hyperparameter that gives the minimum rmse score.

# First we will use GridSearchCV for the RandomForestRegressor with the following list of hyperparameters.

# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
 scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


# The best parameters are

# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.best_estimator_


# Viewing the score for all combination of parameters that were used for testing the model.

# In[ ]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
     print(np.sqrt(-mean_score), params)


# # Prediction

# Predicting the median_house_value using the best estimators in the gridsearch.

# In[ ]:


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) 


# The final rmse score for the test set is 47572.

# In[ ]:


final_rmse

