#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import required libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
import time
from geopy.distance import great_circle

from collections import Counter
import re
import xgboost as xgb

from sklearn.linear_model import LinearRegression as lg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Load the dataset
data_initial = pd.read_csv('../input/listings_summary.csv')
# Print the columns of Initial Dataset loaded
data_initial.columns


# In[ ]:


# Move the selected Features for analysis into a variable
features_to_keep = ['id', 'space', 'description', 'host_has_profile_pic',
                    'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',  
                   'bedrooms', 'bed_type', 'amenities', 'square_feet', 'price', 'cleaning_fee', 
                   'security_deposit', 'extra_people', 'guests_included', 'minimum_nights',  
                   'instant_bookable', 'cancellation_policy', 'experiences_offered', 
                    'neighborhood_overview','access', 'house_rules']

# Load the Features into a dataset variable
data_raw = data_initial[features_to_keep].set_index('id')
# Check the Shape of the Dataset
print("The dataset with selected features has {} rows and {} columns.".format(*data_raw.shape))


# Clean, Normalize and Standardize the Features

# In[ ]:


# Normalizing the 'room_type' feature
data_raw.room_type.value_counts(normalize=True)


# In[ ]:


# Normalizing the 'property_type' feature
data_raw.property_type.value_counts(normalize=True)


# In[ ]:


#Print First 3 rows of the selected features
data_raw[['price', 'cleaning_fee', 'extra_people', 'security_deposit']].head(3)


# In[ ]:


# Checking for Nan's in 'price' column
data_raw.price.isna().sum()


# In[ ]:


# Checking for Nan's in 'cleaning_fee' column
data_raw.cleaning_fee.isna().sum()


# In[ ]:


#Replace Nan's with $0.00 for 'cleaning_fee'
data_raw.cleaning_fee.fillna('$0.00', inplace=True)
data_raw.cleaning_fee.isna().sum()


# In[ ]:


# Checking for Nan's in 'security_deposit' column
data_raw.security_deposit.isna().sum()


# In[ ]:


#Replace Nan's with $0.00 for 'security_deposit'
data_raw.security_deposit.fillna('$0.00', inplace=True)
data_raw.security_deposit.isna().sum()


# In[ ]:


# Checking for Nan's in 'extra_people' column
data_raw.extra_people.isna().sum()


# In[ ]:


# Cleaning up the features using method chaining
data_raw.price = data_raw.price.str.replace('$', '').str.replace(',', '').astype(float)
data_raw.cleaning_fee = data_raw.cleaning_fee.str.replace('$', '').str.replace(',', '').astype(float)
data_raw.security_deposit = data_raw.security_deposit.str.replace('$', '').str.replace(',', '').astype(float)
data_raw.extra_people = data_raw.extra_people.str.replace('$', '').str.replace(',', '').astype(float)


# In[ ]:


# Analyzing the 'price' feature
data_raw['price'].describe()


# In[ ]:


green_square = dict(markerfacecolor='g', markeredgecolor='g', marker='.')
data_raw['price'].plot(kind='box', xlim=(0, 1000), vert=False, flierprops=green_square, figsize=(16,3));


# In[ ]:


# Based on the plot, to improve the dataset quality removed listings with prices above 400 and 0.00 
data_raw.drop(data_raw[ (data_raw.price > 400) | (data_raw.price == 0) ].index, axis=0, inplace=True)
data_raw['price'].describe()
print("The dataset after price-wise preprocessed has {} rows and {} columns.".format(*data_raw.shape))


# In[ ]:


# Viewing all the dataset features for Nan's and Missing Values
data_raw.isna().sum()


# In[ ]:


# Droping features with too many Nan's
data_raw.drop(columns=['square_feet', 'space','neighborhood_overview','access', 'house_rules'], inplace=True)


# In[ ]:


# Droping rows with Nan's in features 'bathrooms', 'bedrooms'
data_raw.dropna(subset=['bathrooms', 'bedrooms', ], inplace=True)


# In[ ]:


# Replacing Nan's with no for 'host_has_profile_pic' 
data_raw.host_has_profile_pic.fillna(value='f', inplace=True)
data_raw.host_has_profile_pic.unique()


# In[ ]:


# Checking the dataset features after dropping features with higher Nan's
data_raw.isna().sum()


# In[ ]:


# 'description' has lot of Nan's yet there might be useful information regading size
# Trying to extract size by identyfying numbers followed by text like 'sm' or 'm' and 
# transform it into a new feature 'size'
# Extract numbers from 'description' feature
data_raw['size'] = data_raw['description'].str.extract('(\d{2,3}\s?[smSM])', expand=True)
data_raw['size'] = data_raw['size'].str.replace("\D", "")

# Now change datatype of size into float
data_raw['size'] = data_raw['size'].astype(float)


# In[ ]:


# Dropping 'description' feature
data_raw.drop(['description'], axis=1, inplace=True)


# In[ ]:


#Adding a new feature 'distance' to the dataset as location is most important in determining the price
def distance_from_midberlin(lat, lon):
    berlin_centre = (52.5027778, 13.404166666666667)
    record = (lat, lon)
    return great_circle(berlin_centre, record).km

data_raw['distance'] = data_raw.apply(lambda x: distance_from_midberlin(x.latitude, x.longitude), axis=1)


# In[ ]:


print("After preprocessing for missing values and adding new features the dataset has {} rows and {} columns.".format(*data_raw.shape))


# Using Linear Regressor Model for Training the data to predict missing values for 'size'

# In[ ]:


# Filtering out sub_data to detrmine missing values in 'size' based on related independent features
sub_data = data_raw[['accommodates', 'bathrooms', 'bedrooms',  'price', 'cleaning_fee', 
                 'security_deposit', 'extra_people', 'guests_included', 'distance', 'size']]


# In[ ]:


# Split datasets into train and test
train_data = sub_data[sub_data['size'].notnull()]
test_data  = sub_data[sub_data['size'].isnull()]

# Define X
X_train = train_data.drop('size', axis=1)
X_test  = test_data.drop('size', axis=1)

# Define y
y_train = train_data['size']


# In[ ]:


# Describe train_data, test_data, X_train, x_test and y_train data sets 
print("Shape of Train Data:    ",train_data.shape)
print("Shape of Test Data:    ",test_data.shape)
print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("\nShape of y_train:", y_train.shape)


# In[ ]:



# instantiate the Linear Regression Model
linreg = lg()

# Fit Linear regression model to training data
linreg.fit(X_train, y_train)


# In[ ]:


# Make predictions using the model
y_test = linreg.predict(X_test)


# In[ ]:


# Add the new predicted values to the dataframe 'size' feature
y_test = pd.DataFrame(y_test)
y_test.columns = ['size']
print(y_test.shape)
y_test.head()


# In[ ]:


print(X_test.shape)
X_test.head()


# In[ ]:


# Add the index of X_test to an own dataframe
prelim_index = pd.DataFrame(X_test.index)
prelim_index.columns = ['prelim']

# Concat this dataframe with y_test to form our new test dataset
y_test = pd.concat([y_test, prelim_index], axis=1)
y_test.set_index(['prelim'], inplace=True)
y_test.head()
new_test_data = pd.concat([X_test, y_test], axis=1)


# In[ ]:


# check the new test dataset for Nan's in 'size' feature
print(new_test_data.shape)
new_test_data['size'].isna().sum()


# In[ ]:


# concant train and test data to a new sub dataset
sub_data_new = pd.concat([new_test_data, train_data], axis=0)

print(sub_data_new.shape)
sub_data_new.head()


# In[ ]:


# check if the new sub dataset had Nan's in 'size' column
sub_data_new['size'].isna().sum()


# In[ ]:


# prepare the features before concatening
data_raw.drop(['accommodates', 'bathrooms', 'bedrooms', 'price', 'cleaning_fee', 
             'security_deposit', 'extra_people', 'guests_included', 'distance', 'size'], 
            axis=1, inplace=True)


# In[ ]:


# concate the dataset output from linear regression model to complete original dataframe
df = pd.concat([sub_data_new, data_raw], axis=1)

print(df.shape)


# In[ ]:


# analyze 'size' feature to improve quality of the data
green_square = dict(markerfacecolor='g', markeredgecolor='g', marker='.')
df['size'].plot(kind='box', xlim=(0, 1000), vert=False, flierprops=green_square, figsize=(16,4));


# In[ ]:


# drop the rows with 'size' column values 0 and greater than 400 as they are only few 
df.drop(df[ (df['size'] == 0.) | (df['size'] > 400.) ].index, axis=0, inplace=True)
print("The dataset after preprocessing 'size' feature has {} rows and {} columns.".format(*df.shape))


# In[ ]:


# Analyzing another important feature 'ameneties' 
results = Counter()
df['amenities'].str.strip('{}')               .str.replace('"', '')               .str.lstrip('\"')               .str.rstrip('\"')               .str.split(',')               .apply(results.update)

results.most_common(30)


# In[ ]:


# create a new sub dataframe with 'amenity' and 'count'
sub_df = pd.DataFrame(results.most_common(30), columns=['amenity', 'count'])


# In[ ]:


# ploting the top 20 amenities 
sub_df.sort_values(by=['count'], ascending=True).plot(kind='barh', x='amenity', y='count',  
                                                      figsize=(10,10), legend=False, color='green',
                                                      title='Feature_Amenities')
plt.xlabel('Count');


# In[ ]:


# adding new features using 'amenities'
df['Laptop_friendly_workspace'] = df['amenities'].str.contains('Laptop friendly workspace')
df['TV'] = df['amenities'].str.contains('TV')
df['Family_kid_friendly'] = df['amenities'].str.contains('Family/kid friendly')
df['Host_greets_you'] = df['amenities'].str.contains('Host greets you')
df['Smoking_allowed'] = df['amenities'].str.contains('Smoking allowed')


# In[ ]:


# after adding new features drop redundant 'amenities' column 
df.drop(['amenities'], axis=1, inplace=True)


# In[ ]:


# Check the exisiting columns on the dataset
df.columns


# In[ ]:


# print information of the dataset 
df.info()


# In[ ]:


# drop the unhelpful columns
df.drop(['latitude', 'longitude', 'property_type'], axis=1, inplace=True)


# In[ ]:


# convert all string columns into categorical features
for col in ['host_has_profile_pic', 'room_type', 'bed_type', 'instant_bookable', 
            'cancellation_policy']:
    df[col] = df[col].astype('category')


# In[ ]:


# define target
target = df[["price"]]

# define features 
features = df.drop(["price"], axis=1)


# In[ ]:


# identify neumerical features
num_fea = features.select_dtypes(include=['float64', 'int64', 'bool']).copy()

# one-hot encoding of categorical features
cat_fea = features.select_dtypes(include=['category']).copy()
cat_fea = pd.get_dummies(cat_fea)


# In[ ]:


# concat numerical and categorical features
features_recoded = pd.concat([num_fea, cat_fea], axis=1)


# In[ ]:


print(features_recoded.shape)
features_recoded.head(2)


# **XGBoost Regressor**

# In[ ]:


# split dataset into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(features_recoded, target, test_size=0.2)


# In[ ]:


# Instantiating XGB Regressor

booster_start = time.time()

booster = xgb.XGBRegressor()

# create Grid
                
param_grid = {'n_estimators': [200, 300, 400],
              'learning_rate': [.03, 0.05, .07], 
              'max_depth': [3, 4, 5],
              'min_child_weight': [4],
              'colsample_bytree': [0.7, 0.8, 1],
              'gamma': [0.0, 0.1, 0.2]}

# instantiate the tuned random forest
booster_grid_search = GridSearchCV(booster, param_grid, cv=3, n_jobs=-1)

# train the tuned random forest
booster_grid_search.fit(X_train, y_train)

# print best estimator parameters found during the grid search
print(booster_grid_search.best_params_)

booster = xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.2, learning_rate=0.1, 
                           max_depth=6, n_estimators=200)
booster.fit(X_train, y_train)
training_preds_booster = booster.predict(X_train)
val_preds_booster = booster.predict(X_test)

booster_end = time.time()

# Printing the results

print("\nTraining RMSE:", round(np.sqrt(mean_squared_error(y_train, training_preds_booster)),4))
print("Validation RMSE:", round(np.sqrt(mean_squared_error(y_test, val_preds_booster)),4))
print("\nTraining r2:", round(r2_score(y_train, training_preds_booster),4))
print("Validation r2:", round(r2_score(y_test, val_preds_booster),4))
print(f"Time taken to run: {round((booster_end - booster_start)/60,1)} minutes")

# Producing a dataframe of feature importances
ft_weights_booster = pd.DataFrame(booster.feature_importances_, columns=['weight'], index=features_recoded.columns)
ft_weights_booster.sort_values('weight', inplace=True)
ft_weights_booster


# In[ ]:


# Plotting feature importances
plt.figure(figsize=(8,20))
plt.barh(ft_weights_booster.index, ft_weights_booster.weight, align='center') 
plt.title("Feature importances in the XGBoost model", fontsize=14)
plt.xlabel("Feature importance")
plt.margins(y=0.01)
plt.show()


# In[ ]:


# Using Cross Validation with 10 fold 
xg_train = xgb.DMatrix(data=X_train, label=y_train)

params = {'colsample_bytree': 0.3,'learning_rate': 0.2,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=xg_train, params=params, nfold=10,
                    num_boost_round=200,early_stopping_rounds=5,metrics="rmse", as_pandas=True, seed=123)

print((cv_results["test-rmse-mean"]).tail(1))


# In[ ]:


# plot the important features needed for Simple Model
feat_importances = pd.Series(booster.feature_importances_, index=features_recoded.columns)
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='green', figsize=(10,5))
plt.xlabel('Relative Feature Importance with XGBoost');


# In[ ]:


# Print the columns of original dataset
features_recoded.columns


# In[ ]:


# Define Simple Model Dataset with 10 features
simplefeatures_to_keep = ['accommodates', 'bathrooms', 'bedrooms', 'extra_people', 'guests_included', 'size',
                          'room_type_Private room','room_type_Entire home/apt','room_type_Shared room', 
                          'cancellation_policy_super_strict_60']
simplefeatures = features_recoded[simplefeatures_to_keep]

# Check the Shape of the Dataset with important features for Simple Model
print("The dataset with selected features has {} rows and {} columns.".format(*simplefeatures.shape))

# split dataset into training and test datasets
SimpleX_train, SimpleX_test, Simpley_train, Simpley_test = train_test_split(simplefeatures, target, test_size=0.2)

# Instantiating XGB Regressor for training Simple model

simplebooster_start = time.time()

simplebooster = xgb.XGBRegressor()

simplebooster = xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.2, learning_rate=0.1, 
                           max_depth=6, n_estimators=200, random_state=4)
simplebooster.fit(SimpleX_train, Simpley_train)
training_preds_simplebooster = simplebooster.predict(SimpleX_train)
val_preds_simplebooster = simplebooster.predict(SimpleX_test)

simplebooster_end = time.time()


# Printing the results for Simple Model

print("\nTraining RMSE:", round(np.sqrt(mean_squared_error(Simpley_train, training_preds_simplebooster)),4))
print("Validation RMSE:", round(np.sqrt(mean_squared_error(Simpley_test, val_preds_simplebooster)),4))
print("\nTraining r2:", round(r2_score(Simpley_train, training_preds_simplebooster),4))
print("Validation r2:", round(r2_score(Simpley_test, val_preds_simplebooster),4))
print(f"Time taken to run: {round((simplebooster_end - simplebooster_start)/60,1)} minutes")


# **Decision Tree Regressor Model**

# In[ ]:


# split test and train datasets 
X = features_recoded
y = df["price"]

X_treetrain, X_treetest, y_treetrain, y_treetest = train_test_split(X, y, test_size=0.2,random_state=42)


# In[ ]:


# instantiate Decision Tree Regression Model

tree_start = time.time()

regr_tree = DecisionTreeRegressor()
# fit and train model
regr_tree.fit(X_treetrain, y_treetrain)

val_pred_decisiontree = regr_tree.predict(X_treetest)

tree_end = time.time()

# Printing the results for Decision Tree Regression Model

print("Validation RMSE:", round(np.sqrt(mean_squared_error(y_treetest, val_pred_decisiontree)),4))
print(f"Time taken to run: {round((tree_end - tree_start),1)} seconds")

