import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Read the data
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

# Specify Target(y) and Predictors(X)
train_target = train.SalePrice
train_predictors= train.drop(['SalePrice'], axis=1)
train_numeric_predictors = train_predictors.select_dtypes(exclude=['object'])
test_numeric_predictors = test.select_dtypes(exclude=['object'])

# Function
X_train, X_test, y_train, y_test = train_test_split(train_numeric_predictors, 
                                                    train_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


########### Cleaning step 1 ########### 


# Missing values removed
cols_with_missing = [col for col in X_train.columns
                        if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))

# Prepare real test dataset
cols_with_missing_testcols = [col for col in test_numeric_predictors.columns
                        if test_numeric_predictors[col].isnull().any()]
#X_test_non_missing = test_numeric_predictors.drop(cols_with_missing, axis=1)
reduced_X_train_non_missing_testcols = X_train.drop(cols_with_missing_testcols, axis=1)
X_test_non_missing_testcols = test_numeric_predictors.drop(cols_with_missing_testcols, axis=1)
# Use the model to make predictions
model = RandomForestRegressor()
model.fit(reduced_X_train_non_missing_testcols, y_train)

# We will look at the predicted prices to ensure we have something sensible.
#predicted_prices_m1 = model.predict(X_test_non_missing)
predicted_prices_m2 = model.predict(X_test_non_missing_testcols)
# Prepare Submission File
#my_submission_m1 = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices_m1})
my_submission_m2 = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices_m2})
# you could use any filename. We choose submission here
#my_submission_m1.to_csv('submission_missing_m1.csv', index=False)
my_submission_m2.to_csv('submission_missing_m2.csv', index=False)


########### Cleaning step 2 ########### 


# Imputation done for missing values
from sklearn.preprocessing import Imputer

my_imputer = Imputer()

Imputed_X_train = my_imputer.fit_transform(X_train)
Imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(Imputed_X_train, Imputed_X_test, y_train, y_test))

# Use the model to make predictions
model = RandomForestRegressor()
model.fit(Imputed_X_train, y_train)
Imputed_X_test_i1 = my_imputer.transform(test_numeric_predictors)

# We will look at the predicted prices to ensure we have something sensible.
predicted_prices_i1 = model.predict(Imputed_X_test_i1)

# Prepare Submission File
my_submission_i1 = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices_i1})

# you could use any filename. We choose submission here
my_submission_i1.to_csv('submission_missing_i1.csv', index=False)


########### Cleaning step 3 ########### 


# Imputation done for missing values with extra columns showing which one were imputed
X_train_plus = X_train.copy()
X_test_plus = X_test.copy()

for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()     
    X_test_plus[col + '_was_missing'] = X_test_plus[col].isnull()     

Imputed_X_train_plus = my_imputer.fit_transform(X_train_plus)
Imputed_X_test_plus = my_imputer.transform(X_test_plus)
print("Mean Absolute Error from Imputation with missing indicators:")
print(score_dataset(Imputed_X_train_plus, Imputed_X_test_plus, y_train, y_test))

# Prepare real test dataset
X_test_plus_final=test_numeric_predictors.copy()
for col in cols_with_missing:
    X_test_plus_final[col + '_was_missing'] = test_numeric_predictors[col].isnull() 
Imputed_X_test_plus_final = my_imputer.transform(X_test_plus_final)
# Use the model to make predictions
model = RandomForestRegressor()
model.fit(Imputed_X_train_plus, y_train)

# We will look at the predicted prices to ensure we have something sensible.
predicted_prices_i2 = model.predict(Imputed_X_test_plus_final)

# Prepare Submission File
my_submission_i2 = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices_i2})
# you could use any filename. We choose submission here
my_submission_i2.to_csv('submission_imputed_i2.csv', index=False)