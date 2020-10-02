# ----------------------------------------------Background for using XGBoost----------------------------------------------
# 1. Speed & Performance due to originally written in C++
# 2. Parallel computing: Multicore CPU and GPU, which makes feasible to apply it with very large datasets
# 3. Outperforms often benchmarks, 
# 4. Wide variety to tune the method

# Import Boston: dataset
from sklearn.datasets import load_boston
boston = load_boston()

# Dataset info:
print(boston.keys())
print(boston.data.shape)
print(boston.feature_names)
print(boston.DESCR)

# Put data into dataframe and label column headers:
import pandas as pd
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

# See first five rows of the data
data.head()

# Info about each column data e.g. missing data: 506 rows and 14 columns. Describe tells more statistics
data['PRICE'] = boston.target
data.info()
data.describe()

# Think if data includes categorical features to first encode data e.g. with one-hot-encoder
# if missing data, you may want to do something with them, altough XGBoost is capable of handling missing values
# Check any other tasks needed, before training with the XGBoost

# Install XGBOOS library:
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Separate targetvariable and the rest of the variables
X, y = data.iloc[:, :-1],data.iloc[:,-1]

# Convert to optimized data-structure, which XGBoost supports
data_dmatrix = xgb.DMatrix(data=X,label=y)

# create train and test set for cross-validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Initiate XGBoostRegressor object with hyperparameters passed as atguments. 
# In case was instead classification problem, then would use instead: XGBClassifier() class
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators =10)

# Fit the regressor to the training set with fit()
xg_reg.fit(X_train,y_train)

# Make predictions to the test set with predict()
preds = xg_reg.predict(X_test)

# Compute the rmse from sklearns metrics module imported earlier
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# Below the crossvalidation was performed on entire dataset
# However better is first perform k-fold cross validation on training data
# After review its performance  on test dataset

#-------k-fold Cross Validation using XGBoost-------
# XGBoost supports the k-fold cross validation with the cv() method
# nfolds is number of cross-validation sets to be build
# More parameters in XGBoost API reference: https://xgboost.readthedocs.io/en/latest/python/python_api.html

#Create Hyper Parameter dictionary params and exclude n_estimators and include num_boost_rounds
params = {"objective":"reg:linear",'colsample_bytree': 0.3, 'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}

# nfold is 3, so three round cross validation set using XGBoost cv() 
# cv_results include the train and test RMSE metrics for each round
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
cv_results.head()

# Print Final boosting round metrics
# The final result may depend upon the technique used, so you may want to try different
# e.g. grid search, random search Bayesian optimization
print((cv_results["test-rmse-mean"]).tail(1))

# Train the XGBoost 
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

#plot the data and may require install of pip install graphviz package
import matplotlib.pyplot as plt
xgb.plot_tree(xg_reg, num_trees=0)
plt.rcParams['figure.figsize'] = [50,10]
plt.show()

# Alternative is to plot importance of each feature with features ordered as many time as appear
# The plot allows thus to select features for the model
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] =[5,5]
plt.show()

#The idea of this code is to go through the example of the: https://www.datacamp.com/community/tutorials/xgboost-in-python guideline for practice purposes

