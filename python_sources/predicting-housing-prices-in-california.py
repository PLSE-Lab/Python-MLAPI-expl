#!/usr/bin/env python
# coding: utf-8

# # Predicting Housing Prices in California

# ## Table of Contents
# 
# 1. [Loading the Data](#loading-the-data)
#     * [Setup](#setup)
#     * [Data](#data)
# 2. [Understanding the Data](#understanding-the-data)
#     * [Conclusions](#conclusions)
# 3. [Data Preparation](#data-preparation)
#     * [Cleaning up bad values](#cleaning-up-bad-values)
#     * [Separating values and labels](#separating-values-and-labels)
#     * [Feature engineering](#feature-engineering)
#     * [Splitting up the dataset](#splitting-up-the-dataset)
#     * [Normalization](#normalization)
# 4. [Machine Learning](#data-preparation-and-machine-learning)
#     * [Setting up the model](#setting-up-the-model)
#     * [Training the model](#training-the-model)
#     * [Testing the model](#testing-the-model)
# 4. [Making a Benchmark Submission](#making-a-benchmark-submission)

# **Note from competition**
# 
# Here's a simple getting started notebook that shows you how to load the data, and how to create a Kaggle submission file. Remember that you should structure your notebook after the 8 step guide, as detailed in the [Assignment 1 instructions](https://hvl.instructure.com/courses/9086/assignments/17277). 

# ## Loading the Data <a class="anchor" id="loading-the-data"></a>
# 
# Before we do anything, we need to make sure we have all our data ready.

# Importing Numpy now so that it is ready for later.

# In[ ]:


import numpy as np

# Set the random seed for reproducability
np.random.seed(42)


# We will use Pandas throughout the notebook to hold and manage our datasets.

# In[ ]:


import pandas as pd


# Go to Kaggle competition website and download the data. Make a new folder in your DAT158ML repository called `data`. Store the Kaggle competition data in this folder.

# Then you should uncomment the code and run the following two cells. **Warning:** This doesn't work in this Kaggle hosted notebook! See below

# In[ ]:


# Reads in the csv-files and creates a dataframe using pandas

# base_set = pd.read_csv('data/housing_data.csv')
# benchmark = pd.read_csv('data/housing_test_data.csv')
# sampleSubmission = pd.read_csv('data/sample_submission.csv')


# #### Kaggle-specific way of accessing the data
# 
# On Kaggle the data is stored in the folder `../input/dat158-2019/`.

# In[ ]:


base_set = pd.read_csv('../input/dat158-2019/housing_data.csv')
benchmark = pd.read_csv('../input/dat158-2019/housing_test_data.csv')
sample_submission = pd.read_csv('../input/dat158-2019/sample_submission.csv')


# ## Understanding the data
# 
# Now that we have our data, we need to investigate it so that we are able to leverage it to the fullest extent.

# We will use Matplotlib to plot various things throughout the notebook.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


base_set.head()


# In[ ]:


benchmark.head()


# In[ ]:


base_set.info()


# In[ ]:


benchmark.info()


# In[ ]:


base_set.describe()


# Looking at the correlations between the values, we can see that the median income has the strongest correlation to the median house value.

# In[ ]:


correlations = base_set.corr()
correlations["median_house_value"]


# In[ ]:


base_set.hist(bins=50, figsize=(15,15))
plt.show()


# Scatter plot showing the distribution of housing value across California, from low (blue) to high (red).

# In[ ]:


base_set.plot(kind="scatter", 
           x="longitude", 
           y="latitude", 
           alpha=0.4,
           s=base_set["population"]/100, 
           label="population",
           c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
           figsize=(15,7))
plt.legend()


# In[ ]:


from pandas.plotting import scatter_matrix
attributes = ['median_house_value', 'median_income',
             'total_rooms', 'housing_median_age']
scatter_matrix(base_set[attributes], figsize=(12,8))


# ### Conclusions
# 
# ...

# ## Data preparation
# 
# In this section we will preprocess the data and construct a model, which we will then train so we are able to make predictions with it. Lastly we will test in on a test set we create.

# **Note from competition**
# 
# This part you should code and figure out yourself. Play around with different ways to prepare the data, different machine learning models and settings of hyperparameters
# 
# Remember to create your own validation set to evaluate your models. Your test set will not contain labels and are therefore not suited for evaluating and tuning your different models.

# ### Cleaning up bad values
# 
# In this section we handle empty values in the dataset. There are `null` values in both the main dataset and the benchmark set, which we have to clean up. Two of the most common possible strategies for this are as follows:
# 
# 1. Remove rows with `NaN` values
# 2. Fill `NaN` values with median or mean of all the other values
# 
# Here, we go for the latter, because we want to use all the data as best as we can. Median is good for values that vary a lot, so we go for that here.

# In[ ]:


# There are null values in total_bedrooms, we fill those with the median
def fill_null(dataset, column):
    values = {column: dataset[column].median()}
    
    return dataset.fillna(values)

# For these particular sets, there are only null values in the 'total_bedrooms' column.
base_set = fill_null(base_set, 'total_bedrooms')
benchmark = fill_null(benchmark, 'total_bedrooms')


# The `Id` column in the benchmark set is not needed, so we remove that.

# In[ ]:


benchmark = benchmark.drop(columns=['Id'])


# Finally, we check to see that neither of the datasets contain `NaN` values.

# In[ ]:


base_set.isnull().any()


# In[ ]:


benchmark.isnull().any()


# ### Separating values and labels
# 
# It is time to split the dataset into values and labels. To do that, we drop the label column and call that `X`, and take the label column alone and call that `Y`. Afterwards we are ready to start shaping our dataset.

# In[ ]:


labels_column = 'median_house_value'

X = base_set.drop(columns=[labels_column])
Y = pd.DataFrame(base_set[labels_column], columns=[labels_column])


# In[ ]:


X.head()


# In[ ]:


Y.head()


# ### Feature engineering
# 
# Here we derive useful datapoints from existing ones. We also use one-hot encoding to transform string columns into separate columns containing numbers.

# In[ ]:


def derive_datapoints(dataset):
    dataset['bedrooms_per_room'] = dataset['total_bedrooms'] / dataset['total_rooms']

    dataset['rooms_per_household'] = dataset['total_rooms'] / dataset['households']
    dataset['bedrooms_per_household'] = dataset['total_bedrooms'] / dataset['households']
    dataset['population_per_household'] = dataset['population'] / dataset['households']
    
    return dataset

X = derive_datapoints(X)
benchmark = derive_datapoints(benchmark)

# One-hot encoding
X = pd.get_dummies(X)
benchmark = pd.get_dummies(benchmark)


# For this particular dataset, we also need to do a couple extra things to the benchmark set.

# In[ ]:


# Some housekeeping, we need to ensure the test set has the same columns as the training set
# The missing columns will be the onehot-encoded values
missing_columns = set( X.columns ) - set( benchmark.columns )

# We fill the values in the missing columns with 0, as they are one-hot encoded values that don't exist in the set
for column in missing_columns:
    benchmark[column] = 0

# Ensure the order of column in the test set is in the same order than in train set
benchmark = benchmark[X.columns]


# In[ ]:


X.head()


# In[ ]:


benchmark.head()


# ### Splitting up the dataset
# 
# We split our base set into separate datasets for training, testing and validation.

# In[ ]:


from sklearn.model_selection import train_test_split

train_to_valtest_ratio = .2
validate_to_test_ratio = .5

# First split our main set
(X_train,
 X_validation_and_test,
 Y_train,
 Y_validation_and_test) = train_test_split(X, Y, test_size=train_to_valtest_ratio)

# Then split our second set into validation and test
(X_validation,
 X_test,
 Y_validation,
 Y_test) = train_test_split(X_validation_and_test, Y_validation_and_test, test_size=validate_to_test_ratio)


# ### Normalization

# To make the scales of the numbers appropriate for the neural network, we should to do some scaling.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler

X_scaler = StandardScaler().fit(X_train)
def scale_dataset_X(dataset):
    return X_scaler.transform(dataset)

X_train_scaled = scale_dataset_X(X_train)
X_validation_scaled = scale_dataset_X(X_validation)
X_test_scaled = scale_dataset_X(X_test)


# ### Machine Learning

# ### Set up the model
# 
# Now, it is time to set up the architecture.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(15, activation='relu', input_dim=X_train.shape[1]),
    Dense(15, activation='relu'),
    Dense(60, activation='relu'),
    Dense(120, activation='relu'),
    Dense(60, activation='relu'),
    Dense(15, activation='relu'),
    Dense(1),
])

model.summary()


# In[ ]:


import keras.backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

model.compile(optimizer='adadelta', # adam, sgd, adadelta
              loss=rmse,
              metrics=[rmse, 'mae'])


# ### Training the model
# 
# Let's fit the model on the data.

# In[ ]:


from keras.callbacks import EarlyStopping

early_stopper = EarlyStopping(patience=3)

training_result = model.fit(X_train_scaled, Y_train,
                            batch_size=32,
                            epochs=250,
                            validation_data=(X_validation_scaled, Y_validation),
                            callbacks=[early_stopper])


# Now, let's look into how the fitting went.

# In[ ]:


# Plot model accuracy over epoch
plt.plot(training_result.history['mean_absolute_error'])
plt.plot(training_result.history['val_mean_absolute_error'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Plot model loss over epoch
plt.plot(training_result.history['loss'])
plt.plot(training_result.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[ ]:


validate_result = model.test_on_batch(X_validation_scaled, Y_validation)
validate_result


# ### Testing the model
# 
# Finally, we churn the test set through the model we created.

# In[ ]:


test_result = model.test_on_batch(X_test_scaled, Y_test)
test_result


# ### Trying other models

# Testing with RandomForestRegressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rfr_model = RandomForestRegressor()
rfr_model.fit(X_train, Y_train)

# Get the mean absolute error on the validation data
rfr_predictions = rfr_model.predict(X_test)

rfr_error =  np.sqrt(np.mean((rfr_predictions - Y_test['median_house_value']) ** 2))
rfr_error


# Testing with XGBoost

# In[ ]:


import re

regex = re.compile(r"[|]|<", re.IGNORECASE)

# XGBoost does not support some of the column names

X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]
X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns.values]

from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import scipy.stats as st

one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

xgb_reg = XGBRegressor(nthreads=-1)

xgb_gs_params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}

xgb_gs = RandomizedSearchCV(xgb_reg, xgb_gs_params, n_jobs=1)  
xgb_gs.fit(X_train.as_matrix(), Y_train)  

xgb_model = xgb_gs.best_estimator_ 

xgb_predictions = xgb_model.predict(X_test.as_matrix())

xgb_error =  np.sqrt(np.mean((xgb_predictions - Y_test['median_house_value']) ** 2))
xgb_error


# A comparison of all the models

# In[ ]:


print(f'NN RMSE:                            {test_result[0]}')
print(f'RandomForestRegressor RMSE:         {rfr_error}')
print(f'XGBRegressor RMSE:                  {xgb_error}')


# ## Making a Benchmark Submission
# 
# For the benchmark data, it is important that we put it through the same preparation steps as the training set.

# In[ ]:


# Scale test data
benchmark_scaled = scale_dataset_X(benchmark)


# In[ ]:


benchmark.head()


# In[ ]:


X.head()


# In[ ]:


median_house_value = model.predict(benchmark_scaled)


# **Note from competition**
# 
# After you have trained your model and have found predictions on your test data, you must create a csv-file that contains 'Id' and your predictions in two coloums.
# 
# We have assumed that you have called your predictions 'median_house_value' after you have trained your model.
# 
# This is just for demonstrational purposes, that is why all our predictions is zero. Yours will be filled with numbers.

# In[ ]:


len(median_house_value)


# In[ ]:


median_house_value


# In[ ]:


submission = pd.DataFrame({
    'Id': [i for i in range(len(median_house_value))],
    'median_house_value': median_house_value.flatten()
})


# In[ ]:


submission.head()


# In[ ]:


# Stores a csv file to submit to the kaggle competition
submission.to_csv('submission.csv', index=False)

