import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# splitting data
X_train, X_valid, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# define model
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

# Function for comparing different approaches
from sklearn.metrics import mean_absolute_error
def score_dataset(X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]     

# drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# Encoding Categorical Data using OHE
s = (reduced_X_train.dtypes == 'object')
object_cols = list(s[s].index)
print(object_cols)

from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(reduced_X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(reduced_X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = reduced_X_train.index
OH_cols_valid.index = reduced_X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = reduced_X_train.drop(object_cols, axis=1)
num_X_valid = reduced_X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE after One-Hot Encoding:") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

# final training and validation data
final_X_train = OH_X_train
final_X_valid = OH_X_valid

# Define and fit model
model.fit(final_X_train, y_train, 
            early_stopping_rounds=100,
            eval_set=[(final_X_valid, y_valid)],)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your appraoch):")
print(mean_absolute_error(y_valid, preds_valid))

# Fit model to entire dataset for better predictions
final_X = pd.concat([final_X_train, final_X_valid], axis=0)
final_y = pd.concat([y_train, y_valid], axis=0)
model = XGBRegressor(n_estimators=733, learning_rate=0.05, n_jobs=4)
model.fit(final_X, final_y)

# preprocess test data
reduced_X_test = X_test_full.drop(cols_with_missing, axis=1)
print(reduced_X_test.head())

reduced_X_test.fillna('NaN', inplace=True)

OH_cols_test = pd.DataFrame(OH_encoder.transform(reduced_X_test[object_cols]))
OH_cols_test.index = reduced_X_test.index

num_X_test = reduced_X_test.drop(object_cols, axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()

final_X_test = pd.DataFrame(imputer.fit_transform(OH_X_test))
final_X_test.columns = OH_X_test.columns
final_X_test.index = OH_X_test.index

print(final_X_test.head())

# get test predictions
preds_test = model.predict(final_X_test)
print(preds_test.shape)


# Save test predictions to file
output = pd.DataFrame({'Id': final_X_test.index,
                       'SalePrice': preds_test})
output.head()
output.to_csv('submission.csv', index=False)