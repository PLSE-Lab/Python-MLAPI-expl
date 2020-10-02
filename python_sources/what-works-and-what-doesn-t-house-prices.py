#!/usr/bin/env python
# coding: utf-8

# # House Prices Competition
# 
# The purpose of this competition is to practice different regression modelling techniques and data pre-processing approaches.

# ## Import Relevant Packages

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing # package containing modules for modelling, data processing, etc.
from sklearn import impute
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
import seaborn as sns # visualization package #1
import matplotlib.pyplot as plt # visualization package #2
# Configure visualisations
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import data set .csv files and transform to dataframes

# In[ ]:


# Import files containing data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Convert .csv to dataframes
train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", index_col="Id")
test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv", index_col="Id")


# # Define a Simple Linear Regressor - Baseline Model
# 
# Start by defining a simple linear regressor model to use as baseline. The LabelEncoder module shall be used to encode categorical feaatures. Moreover, the SimpleImputer module imputing median values will be used to fill not-a-number entries.
# 

# In[ ]:


# concatenate both train and test sets
# create extra column to allow a later separation of test and train data sets
train_df['isTrain'] = 1
test_df['isTrain'] = 0
dataset = pd.concat([train_df, test_df], axis=0, sort=False)

# fillna, substitutes nan values with mean values in that feature
dataset.fillna(dataset.mean(), inplace=True)

# Fill na's of categorical features with simple imputer
imputer = impute.SimpleImputer(missing_values= np.nan, strategy='constant', fill_value='Empty')
imputed_df = pd.DataFrame(imputer.fit_transform(dataset), index=dataset.index, columns=dataset.columns, dtype=np.array(dataset.dtypes))
# convert imputed_df features to those of train_df
# when creating df - imputed_df - all features are assigned dtype "object"
columns = dataset.select_dtypes(include="int").columns
imputed_df[columns] = imputed_df[columns].astype(int)
columns = dataset.select_dtypes(include="float").columns
imputed_df[columns] = imputed_df[columns].astype(float)
columns = dataset.select_dtypes(include="object").columns
imputed_df[columns] = imputed_df[columns].astype(object)

# Define Ordinal encoder to convert categorical features to numerical
OE_encoder = preprocessing.OrdinalEncoder()
cat_features = imputed_df.select_dtypes(include=['object']).dtypes.index # create array containing indexes of categorical features
imputed_df_num = imputed_df.select_dtypes(exclude=['object']).copy() # create dataframe containing only numerical features
imputed_df_cat = imputed_df[cat_features].copy() # create dataframe containing only categorical features
cat_feat_encoded = pd.DataFrame(OE_encoder.fit_transform(imputed_df_cat), index=imputed_df_cat.index, columns=cat_features) #create dataframe after fitting and transforming categorical features
df_encoded = pd.concat([imputed_df_num, cat_feat_encoded], axis=1) # concatenate numerical and categorical dataframes

# Separate test and train data sets after modifications
train_df2 = df_encoded[df_encoded['isTrain'] == 1].copy()
test_df2 = df_encoded[df_encoded['isTrain'] == 0].copy()

# Drop create feature 'isTrain'
train_df2.drop(['isTrain'], axis=1, inplace=True)
test_df2.drop(['isTrain', 'SalePrice'], axis=1, inplace=True)

# Set objective feature
y = train_df2["SalePrice"]
X = train_df2.drop(["SalePrice"], axis=1).copy()

# Split data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# Use a linear regression model as baseline
Baseline_model = LinearRegression()
Baseline_model.fit(X_train, y_train) #training the algorithm
y_pred = Baseline_model.predict(X_valid) #make predictions on algorithm

# Use a decision tree model as baseline #1
Baseline_model_DT = DecisionTreeRegressor(max_depth=10, min_samples_leaf=14, random_state=0) # max_depth and min_samples_leaf selected after performing small side study
Baseline_model_DT.fit(X_train, y_train) #training the algorithm
y_pred_DT = Baseline_model_DT.predict(X_valid) #make predictions on algorithm

# Accuracy results of Linear Regressor model to check for overfitting
print("Linear Regressor - Train set R2 score is: ", round(Baseline_model.score(X_train, y_train),4))
print("Linear Regressor - Validation set R2 score is: ", round(Baseline_model.score(X_valid, y_valid),4), "\n")

# Accuracy results of Linear Regressor model to check for overfitting
print("Decision Tree Regressor - Train set R2 score is: ", round(Baseline_model_DT.score(X_train, y_train),4))
print("Decision Tree Regressor - Validation set R2 score is: ", round(Baseline_model_DT.score(X_valid, y_valid),4), "\n")

# Compute Root Mean Square Error
print("The RMS log error of Linear Regressor Baseline Model is: ", round(mean_squared_log_error(y_valid, np.absolute(y_pred)),4))
print("The RMS log error of Decision Tree Regressor Baseline Model is: ", round(mean_squared_log_error(y_valid, y_pred_DT),4), "\n")


# The `LinearRegressor` model shows some interesting results! Why is there such a discrepancy between train and validation sets accuracy? Are the distributions of the variables inside each set very different from one another? 

# In[ ]:


# construction of a plot to show differences between original SalePrice and its logarithmic distribution
plt.figure(figsize=(8,6))
sns.distplot(y_train)
sns.distplot(y_valid)


# In[ ]:


# scatter plot showing the differences between the predicted and real SalePrice 
plt.figure(figsize=(9,6))
plt.ylabel('Prediction')
sns.regplot(x=y_valid, y=y_pred, label='Linear Regressor')
sns.regplot(x=y_valid, y=y_pred_DT, label='Decision Tree Regressor')
plt.legend()


# Next, a small study is performed to find good parameters for the Decision Tree Regressor. It is done in the hidden cells.

# In[ ]:


# Side study used to determine best parameters for decision tree regressor
# create 3 empty dictionaries to contain iteration results
results_rmsle = {}
results_accuracy_valid = {}
results_accuracy_train = {}

for depth in range(1,11):
    aux_rmsle = []
    aux_accuracy_train = []
    aux_accuracy_valid = []
    
    for samples in range(4,22,2):
        Baseline_model_DT = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=samples, random_state=0)
        Baseline_model_DT.fit(X_train, y_train) #training the algorithm
        y_pred_DT = Baseline_model_DT.predict(X_valid) #make predictions on algorithm
        
        aux_rmsle.append(round(mean_squared_log_error(y_valid, y_pred_DT),2))
        aux_accuracy_train.append(round(Baseline_model_DT.score(X_train, y_train),2))
        aux_accuracy_valid.append(round(Baseline_model_DT.score(X_valid, y_valid),2))
        
    results_rmsle["depth=" + str(depth)] = aux_rmsle
    results_accuracy_train["depth=" + str(depth)] = aux_accuracy_train
    results_accuracy_valid["depth=" + str(depth)] = aux_accuracy_valid

# convert dictionaries to dataframes
results_rmsle_df = pd.DataFrame(results_rmsle, index=range(4,22,2), columns=results_rmsle.keys())
results_accuracy_train_df = pd.DataFrame(results_accuracy_train, index=range(4,22,2), columns=results_accuracy_train.keys())
results_accuracy_valid_df = pd.DataFrame(results_accuracy_valid, index=range(4,22,2), columns=results_accuracy_valid.keys())

# rename index of dataframes
results_rmsle_df.index.names = ["SamplesPerLeaf"]
results_accuracy_train_df.index.names = ["SamplesPerLeaf"]
results_accuracy_valid_df.index.names = ["SamplesPerLeaf"]

# create 3 plots showing the evolutions of the errors for the different samples per leaf and tree levels
f, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,6))
axes[0].set_title('RMSLE Error')
axes[1].set_title('Accuracy Train Errors')
axes[2].set_title('Accuracy Validation Errors')
sns.lineplot(data=results_rmsle_df, dashes=False, ax=axes[0], legend='brief')
sns.lineplot(data=results_accuracy_train_df, dashes=False, ax=axes[1], legend='brief')
sns.lineplot(data=results_accuracy_valid_df, dashes=False, ax=axes[2], legend='brief')


# As observed, the parameters combination that minimize the errors is:
# * `SamplesPerLeaf` = 14
# * `MaxDepth` = 10

# ## Create Submission file
# 
# Create .csv file with the submissions of the baseline Decision Tree model.

# In[ ]:


# store prediction results from Decision Tree Baseline Model
id = test_df2.index
result = Baseline_model_DT.predict(test_df2)

output = pd.DataFrame( { 'id': id , 'SalePrice': result} )
output = output[['id', 'SalePrice']]

output.to_csv("solution.csv", index = False)
output.head(10)


# # Test #1 - Scaling/Normalization
# Using `sklearn.preprocessing` two modules, A- `MinMaxScaler` and B-`StandardScaler`, scale/normalize all variables to check their influence on the linear regressor baseline model predictions.
# Also, as a further test, apply a logarithmic transformation on predictions and check if results improve.

# In[ ]:


# Use MinMaxScaler module - scales all features between defined feature range (default=(0,1))
mM_scalerX = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(X_train)
mM_scalery = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(np.array(y_train).reshape(-1,1))
X_train_scaled = pd.DataFrame(mM_scalerX.transform(X_train), index=X_train.index, columns=X_train.columns)
y_train_scaled = pd.DataFrame(mM_scalery.transform(np.array(y_train).reshape(-1,1)), index=y_train.index)
X_valid_scaled = pd.DataFrame(mM_scalerX.transform(X_valid), index=X_valid.index, columns=X_valid.columns)
y_valid_scaled = pd.DataFrame(mM_scalery.transform(np.array(y_valid).reshape(-1,1)), index=y_valid.index)

# Use a LinearRegression model - with MinMaxScaler
Scaled_model = LinearRegression()
Scaled_model.fit(X_train_scaled, y_train_scaled) # training the algorithm
y_pred_scaled = mM_scalery.inverse_transform(np.array(Scaled_model.predict(X_valid_scaled)).reshape(-1,1)) # make predictions on algorithm and invert the scaling made

# Use StandardScaler module - standardize features by removing the mean and scaling to unit variance
normalizer_X = preprocessing.StandardScaler().fit(X_train)
normalizer_Y = preprocessing.StandardScaler().fit(np.array(y_train).reshape(-1,1))
X_train_normalized = pd.DataFrame(normalizer_X.transform(X_train), index=X_train.index, columns=X_train.columns)
y_train_normalized = pd.DataFrame(normalizer_Y.transform(np.array(y_train).reshape(-1,1)), index=y_train.index)
X_valid_normalized = pd.DataFrame(normalizer_X.transform(X_valid), index=X_valid.index, columns=X_valid.columns)
y_valid_normalized = pd.DataFrame(normalizer_Y.transform(np.array(y_valid).reshape(-1,1)), index=y_valid.index)

# Use a LinearRegression model - with StandardScaler
Normalized_model = LinearRegression()
Normalized_model.fit(X_train_normalized, y_train_normalized) #training the algorithm
y_pred_normalized = normalizer_Y.inverse_transform(np.array(Normalized_model.predict(X_valid_normalized)).reshape(-1,1)) # make predictions on algorithm and invert the normalization made

# Apply logarithmic transformation on predictions
y_train_log = np.log(y_train)
y_valid_log = np.log(y_valid)

# Use a LinearRegression model - with logarithmic transformation
log_model = LinearRegression()
log_model.fit(X_train, y_train_log) #training the algorithm
y_pred_log = np.exp(log_model.predict(X_valid)) # make predictions on algorithm and invert the log made

# Use a DecisionTreeRegressor model - with logarithmic transformation
log_model_DT = DecisionTreeRegressor(max_depth=10, min_samples_leaf=14, random_state=0)
log_model_DT.fit(X_train, y_train_log) #training the algorithm
y_pred_log_DT = np.exp(log_model_DT.predict(X_valid)) # make predictions on algorithm and invert the log made

# Accuracy results of Linear Regressor model to check for overfitting
print("Linear Regressor with MinMaxScaler - Train set R2 score is: ", round(Scaled_model.score(X_train_scaled, y_train_scaled),4))
print("Linear Regressor with MinMaxScaler - Validation set R2 score is: ", round(Scaled_model.score(X_valid_scaled, y_valid_scaled),4), "\n")

# Accuracy results of Linear Regressor model to check for overfitting
print("Linear Regressor with StandardScaler - Train set R2 score is: ", round(Normalized_model.score(X_train_normalized, y_train_normalized),4))
print("Linear Regressor with StandardScaler - Validation set R2 score is: ", round(Normalized_model.score(X_valid_normalized, y_valid_normalized),4), "\n")

# Accuracy results of Linear Regressor model to check for overfitting
print("Linear Regressor with Log Transformation - Train set R2 score is: ", round(log_model.score(X_train, y_train_log),4))
print("Linear Regressor with Log Transformation - Validation set R2 score is: ", round(log_model.score(X_valid, y_valid_log),4))
# Accuracy results of Linear Regressor model to check for overfitting
print("Decision Tree Regressor with Log Transformation - Train set R2 score is: ", round(log_model_DT.score(X_train, y_train_log),4))
print("Decision Tree Regressor with Log Transformation - Validation set R2 score is: ", round(log_model_DT.score(X_valid, y_valid_log),4), "\n")

# Compute Root Mean Square Error
print("The RMS log error of Linear Model with MinMaxScaler is: ", round(mean_squared_log_error(y_valid, np.absolute(y_pred_scaled)),4))
print("The RMS log error of Linear Model with StandardScaler is: ", round(mean_squared_log_error(y_valid, np.absolute(y_pred_normalized)),4))
print("The RMS log error of Linear Model with log transformation is: ", round(mean_squared_log_error(y_valid, y_pred_log),4))
print("The RMS log error of DT Model with log transformation is: ", round(mean_squared_log_error(y_valid, y_pred_log_DT),4), "\n")


# As observed above, with the log transformation, the `SalePrice` becomes less skewed and more "normal".

# In[ ]:


# construction of a plot to show differences between original SalePrice and its logarithmic distribution
f, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
axes[0].set_title('SalePrice')
axes[1].set_title('log(SalePrice)')
sns.distplot(y_train, ax=axes[0])
sns.distplot(y_train_log, ax=axes[1])


# In[ ]:


f, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
axes[0].set_title('Validation SalePrice')
axes[1].set_title('Predicted e^log(SalePrice)')
sns.distplot(y_valid, ax=axes[0])
sns.distplot(y_pred_log, ax=axes[1])


# # Test #2 - Compute New Features

# In[ ]:


train_df2[train_df2['MasVnrArea'] != 0][['TotalBsmtSF', 'GarageArea', '1stFlrSF','MasVnrArea', 'GrLivArea', 'OpenPorchSF', 'WoodDeckSF', 'EnclosedPorch', 'ScreenPorch', '3SsnPorch']].head(20)


# In[ ]:


train_df2[train_df2['LowQualFinSF'] != 0][['TotalBsmtSF', 'GarageArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'MasVnrArea', 'GrLivArea', 'OpenPorchSF', 'WoodDeckSF', 'EnclosedPorch', 'ScreenPorch', '3SsnPorch']].head(20)


# In[ ]:


# Makes sense to build a single feature containing the entire house square footage of the lot
train_df2['TotalInsideSF'] = train_df2['TotalBsmtSF'] + train_df2['GrLivArea'] +train_df2['GarageArea'] + _
                           train_df2['OpenPorchSF'] + train_df2['WoodDeckSF'] + train_df2['EnclosedPorch'] + _
                           train_df2['ScreenPorch'] + train_df2['3SsnPorch'] + train_df2['PoolArea']
        
train_df2['TotalOutsideSF'] = train_df2['LotArea'] - train_df2['TotalBsmtSF']

# Makes sense to build a single feature containing the entire square footage of the lot
train_df2['TotalHomeSqArea'] = train_df2['TotalBsmtSF'] + train_df2['GrLivArea'] + _
                           train_df2['OpenPorchSF'] + train_df2['WoodDeckSF'] + train_df2['EnclosedPorch'] + _
                           train_df2['ScreenPorch'] + train_df2['3SsnPorch']


# # Test #3 - Feature Correlation
# 
# In this approach, I will select the most correlated features (above a certain threshold) with the `SalePrice` target feature, and use them to build a model. 
# The goal is to check if by removing the least correlated features, hence reducing the associated noise, this first iteration performs better than the baseline model.
# 
# 
# 
# 

# In[ ]:


# Correlation plot - too confuse to understand relationships
#f, axis = plt.subplots(figsize=(12, 12))
#corr = train_df_encoded.corr()
#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=axis)

# compute bar plot showing all correlations between features and SalePrice
corr_SalePrice = train_df2[train_df2.columns[1:]].corr()['SalePrice'][:].sort_values(ascending=False)
corr_SalePrice.drop(["SalePrice"], axis=0, inplace=True)
plt.figure(figsize=(18, 5))
corr_SalePrice.plot.bar()


# ## Possible Paths:
# 1. Try to fill null (NA/nan, etc.) values, with values that make sense for that specific feature. This approach ties to make the most out of every feature available, despite possibly introducing bias to the modelling. 
# 2. Perform some feature engineering, for instance, aggregating all features containing the square footage of a house as `TotalSqFtg`
# 3. Try feature normalization (only relevant in regression type models)
# 4. Try something like an `XGBoost` or `LightGBM` or `ElasticNet`
# 
# Don't discard variables unless you have a good reason for it. No it is not a good reason to say it's not correlated. Tree algorithms can use the information and are not harmed by including it. A good reason for excluting could be I'm using KNN. The other main reason for discarding variables are if they are correlated with being in the test or training set e.g. in many competitions you goal is to predict out of time. So including time can be very harmfull.
