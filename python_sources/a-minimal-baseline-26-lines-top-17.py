#!/usr/bin/env python
# coding: utf-8

# # A minimal baseline for SalePrice prediction
# Most shared kernels show a wide array of data plots and complicated feature engineering, with the implication that all of the complexity will lead to a high-scoring submission. In practice, however, most of it seems to be just guesswork which doesn't necessarily yield exceptional results.
# 
# This kernel does exactly the opposite. It includes the simplest possible handling of missing values and categories, and only uses one bit of feature engineering. Yet, at the time of writing, it is in the top 17% of submissions. (A score of 0.11673, which places it 717 out 4380 on the leaderboard.)
# 
# The specific techniques used are the following:
# * Use the log of the target data (and restore it by exponentiation the final predictions)
# * Replace every missing value with 0.
# * Introduce non-linearity in numeric features by adding the square root of every numeric column (while keeping the original)
# * Use one-hot encoding for categorical features.
# * Eliminate outliers -- defined as any row whose cross-validated prediction error is more than two standard deviations from the mean.
# 
# We avoid using any of the test data in our feature pipeline. It makes no practical difference, but it's just good data science. In particular this approach guarantees that we can handle any new prediction requests which might have unexpected categorical errors.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_predict

rawXy_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
y = np.log(rawXy_train.SalePrice)
rawX_train = rawXy_train.drop(columns=['SalePrice'])
numeric_columns = [col for col in rawX_train.columns if rawX_train[col].dtype != "object"]
def transform_X(X):
    X = X.fillna(0)
    X = X.join(np.sqrt(X[numeric_columns]), rsuffix="_SQRT")
    X = pd.get_dummies(X)
    return X
X_train = transform_X(rawX_train)

cv_errors = np.abs(cross_val_predict(BayesianRidge(), X_train, y, n_jobs=-1, cv=10) - y)
outliers = list(cv_errors[cv_errors > (np.mean(cv_errors) + 2*np.std(cv_errors))].index)
X_train = X_train.drop(outliers)
y = y.drop(outliers)
final_model = BayesianRidge().fit(X_train, y)

X_test = transform_X(pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id'))
X_test = pd.DataFrame({col: X_test.get(col, 0) for col in X_train.columns})
predictions = np.exp(final_model.predict(X_test))
pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions}).to_csv('baseline-submission.csv', index=False)


# ## The code, annotated.
# #### Basic importation
# There are lots of tools that we might use, but these four imports are all that we **need**.
# ~~~
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import BayesianRidge
# from sklearn.model_selection import cross_val_predict
# ~~~
# #### Read and prepare the training data
# This is all basic data preparation. We read the file as a DataFrame, separate out the target column as 'y' and the feature columns as 'rawX_train'. We take the logarithm of the target column since (a) it traditionally yields better results and (b) it reflects the evaluation metric that we are trying to optimize.
# ~~~
# rawXy_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
# y = np.log(rawXy_train.SalePrice)
# rawX_train = rawXy_train.drop(columns=['SalePrice'])
# ~~~
# #### Transform the training data
# We define a method to do this transformation, because we want to make sure that we perform exactly the same transformation on the test data. 
# 
# The two lines that use "fillna" and "get_dummies" are standard Pandas operations, and require little explanation. We take advantage of the fact that "0" will work either as a numeric value or a categorical value. The "SQRT" line creates a new dataframe by applying "sqrt" to all of the numeric columns, and then uses "join" to merge the values back into the train set. The transformed rows have the same names as the originals, so we specify a suffix to avoid name conflicts when joining.
# ~~~
# numeric_columns = [col for col in rawX_train.columns if rawX_train[col].dtype != "object"]
# def transform_X(X):
#     X = X.fillna(0)
#     X = X.join(np.sqrt(X[numeric_columns]), rsuffix="_SQRT")
#     X = pd.get_dummies(X)
#     return X
# X_train = transform_X(rawX_train)
# ~~~
# #### Eliminate outliers from the training data
# We find outliers using the same model that we'll use for our final submission. Using "cross_val_predict" we can determine predictions on the training data *without overfitting*. Subtracting the actual values from the predictions yields the per-row error. (We take the absolute value, since we just want the amount of error rather than its direction.) Given the errors, we can determine the mean and standard deviation of the errors, and use this to find just those errors which are more than two standard deviations beyond the mean. (I picked "2 standard deviations" out of thin air, but it works.)
# 
# "cv_errors" is a Series, which means we can use "index" to get the identities of the outliers. We then simply use "drop" to eliminate these rows from both the train and target data before training our final model.
# 
# Note that we use "BayesianRidge" because it's incredibly fast and works adequately. You can use whatever model you like, and all of the techniques used here will remain valid.
# ~~~
# cv_errors = np.abs(cross_val_predict(BayesianRidge(), X_train, y, n_jobs=-1, cv=10) - y)
# outliers = list(cv_errors[cv_errors > (np.mean(cv_errors) + 2*np.std(cv_errors))].index)
# X_train = X_train.drop(outliers)
# y = y.drop(outliers)
# final_model = BayesianRidge().fit(X_train, y)
# ~~~
# #### Load and transform the test data
# We use pandas to load the data, and our "transform_X" function from above to execute the same transformations that we used on the train data. 
# 
# However, we need one more step: because our categorical data may not have the same set of values, and definitely won't have them in the same order, our one-hot columns won't align with those in the training data. The expression ```pd.DataFrame({col: X_test.get(col, 0) for col in X_train.columns})``` ensures that we have the same columns as the training data, in the same order, and that any missing one-hot columns are filled with "0".
# ~~~
# X_test = transform_X(pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id'))
# X_test = pd.DataFrame({col: X_test.get(col, 0) for col in X_train.columns})
# ~~~
# #### Predict "SalePrice" for the test data and write it to the output file.
# This is all straightforward. However, we have to remember to use "exp" to convert the predictions back into actual dollar amounts.
# ~~~
# predictions = np.exp(final_model.predict(X_test))
# pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions}).to_csv('baseline-submission.csv', index=False)
# ~~~
# #### Profit
