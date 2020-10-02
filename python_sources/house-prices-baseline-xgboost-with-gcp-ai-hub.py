#!/usr/bin/env python
# coding: utf-8

# The notebook below includes a simple example of training a baseline XGBoost model for Kaggle's [competition to predict House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), using the [XGBoost component](https://aihub.cloud.google.com/u/0/p/products%2F0ccd8a63-71a7-4e48-a68b-685692a62e92) from Google Cloud Platform's AI Hub.
# 
# The components on AI Hub make it easy to run models on Google's cloud infrastructure and run distributed training jobs without revising any code. The components also include a visual report that provides a visual inspection of the trained model (see 'Inspect the training job' below).
# 
# To run the notebook locally you'll need to install the following tools:
#  - [Kaggle API](https://github.com/Kaggle/kaggle-api)
#  - [Google Cloud SDK](https://cloud.google.com/sdk/install)
# 
# # Import dependencies and competition data
# 
# Import the Python modules used for this notebook and download the competition data using the Kaggle API.

# In[ ]:


# Import dependencies
import zipfile
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import xgboost as xgb
from IPython.core.display import HTML

# Replace with your GCS bucket
GCS_BUCKET = 'gs://kaggle-xgboost-example'


# In[ ]:


# Import data from Kaggle
# For documentation on using the Kaggle API for Python refer to the official repo: https://github.com/Kaggle/kaggle-api
get_ipython().system('kaggle competitions download -c house-prices-advanced-regression-techniques')

# Unzip the training and test datasets
with zipfile.ZipFile('house-prices-advanced-regression-techniques.zip', 'r') as data_zip:
    data_zip.extractall('data')
# Remove the downloaded compressed file
tf.io.gfile.remove('house-prices-advanced-regression-techniques.zip')


# # Preprocess the data
# 
# Import and preprocess the train and test datasets before training the model. The training and test sets each include 1460 examples. Below I've partitioned 10% of the training data as a validation set.

# In[ ]:


# Import training data
train_data = pd.read_csv('data/train.csv').sample(frac=1)
train_data['set'] = 'train'

# Partition 10% of training data as a validation set
train_data.iloc[0:(int(train_data.shape[0] * 0.1)), train_data.columns.get_loc("set")] = 'validation'

# Import test data
test_data = pd.read_csv('data/test.csv')
test_data['SalePrice'] = None
test_data['set'] = 'test'
# Pull Ids for test dataset for writing submission.csv file
test_ids = test_data['Id']

# Combine training/validation/test sets into single DataFrame
all_data = train_data.append(test_data)
all_data = all_data.drop(labels='Id', axis=1)
# Reorder columns
cols = all_data.columns.tolist()
del cols[-2:]
cols.insert(0, 'SalePrice')
cols.insert(0, 'set')
all_data = all_data[cols]


# ### One-hot encode the categorical features
# 
# The data includes a large number of categorical features. For the sake of simplicity, assume that all integer features are ordinal and perform one-hot encoding for each of the string features.

# In[ ]:


def one_hot_encode_features(features):

    preprocessed_features = pd.DataFrame()
    
    # One-hot encode categorical features
    for col_name in features.columns:
        # Assume that all numeric columns are continuous or ordinal
        if col_name in ['set', 'SalePrice'] or features[col_name].dtype in ['int64', 'float64']:
            preprocessed_features = pd.concat((preprocessed_features, features[col_name]), axis=1)
        else:
            preprocessed_features = pd.concat((preprocessed_features, pd.get_dummies(features[col_name])), axis=1)

    return preprocessed_features


all_data = one_hot_encode_features(all_data)

# Revise column names
col_names = ['set', 'SalePrice']
col_names.extend(['feature_{}'.format(i) for i in range(all_data.shape[1] - 2)])
all_data.columns = col_names

# Split data into train/validation/test sets
train = all_data.loc[all_data['set'] == 'train']
validation = all_data.loc[all_data['set'] == 'validation']
test = all_data.loc[all_data['set'] == 'test']
# Remove 'set' column
train = train.drop('set', axis=1)
validation = validation.drop('set', axis=1)
test = test.drop('set', axis=1)


# ### Impute missing data and scale features
# 
# Replace missing values with the mean of each column from the training data. Then standardize the features by subtracting the column mean and dividing by the column standard deviation.

# In[ ]:


# Pull column-wise mean and standard deviation from training set
train_column_means = train.mean(axis=0)
train_column_sd = train.std(axis=0)

# Imput missing values with column mean
train.iloc[:, 1:] = train.iloc[:, 1:].fillna(train_column_means[1:])
validation.iloc[:, 1:] = validation.iloc[:, 1:].fillna(train_column_means[1:])
test.iloc[:, 1:] = test.iloc[:, 1:].fillna(train_column_means[1:])


# Standardize features
def standardize_features(features, col_means, col_sds):
    for i in range(features.shape[1]):
        if col_sds[i] != 0:
            features.iloc[:, i] = features.iloc[:, i].subtract(col_means[i]).divide(col_sds[i])
    return features


train.iloc[:, 1:] = standardize_features(
    features=train.iloc[:, 1:],
    col_means=train_column_means[1:],
    col_sds=train_column_sd[1:])
validation.iloc[:, 1:] = standardize_features(
    features=validation.iloc[:, 1:],
    col_means=train_column_means[1:],
    col_sds=train_column_sd[1:])
test.iloc[:, 1:] = standardize_features(
    features=test.iloc[:, 1:],
    col_means=train_column_means[1:],
    col_sds=train_column_sd[1:])


# # Write data to CSV files and upload to Google Cloud Storage
# 
# Use [TensorFlow's Gfile class](https://www.tensorflow.org/api_docs/python/tf/io/gfile/GFile) to copy the preprocessed CSV files to a GCS bucket.

# In[ ]:


# Save preprocessed data as CSV files
os.mkdir('data/preprocessed')
train.to_csv('data/preprocessed/train.csv', index=False)
validation.to_csv('data/preprocessed/validation.csv', index=False)
test.to_csv('data/preprocessed/test.csv', index=False)

# Copy the preprocessed CSV data to a GCS bucket
for dataset in tf.io.gfile.glob('data/preprocessed/*.csv'):
    tf.io.gfile.copy(
        dataset,
        os.path.join(GCS_BUCKET, 'house_prices_data', os.path.basename(dataset)),
        overwrite=True)


# # Train an XGBoost model on AI Platform
# 
# To train a model with the XGBoost component on AI Hub, visit the [component's page on AI Hub](https://aihub.cloud.google.com/u/0/p/products%2F0ccd8a63-71a7-4e48-a68b-685692a62e92) and click on the 'Edit Training Command' button. The command that pops up can be copy/pasted and run from the command line.
# 
# Below I've revised this command to work in a Python notebook and filled in parameter values for the training job including location of the training/validation data, the output location to save the trained model, and the hyperparameter values for the training run.
# 
# Running the `gcloud ai-platform jobs submit training` command will submit a training job to the GCP AI Platform Training service. This service will automatically provision the resources required for the job (including one or more machines with optional GPU or TPU devices attached), and run the model training on this cluster.
# 
# Navigate to AI Platform > Jobs page in the [GCP console](https://console.cloud.google.com) to view the logs of the training run.

# In[ ]:


# Set parameter values for a training run
TRAINING_DATA = os.path.join(GCS_BUCKET, 'house_prices_data/train*')
TARGET_COLUMN = 'SalePrice'
VALIDATION_DATA = os.path.join(GCS_BUCKET, 'house_prices_data/val*')
OUTPUT_LOCATION = os.path.join(GCS_BUCKET, 'xgboost_output')
DATA_TYPE = 'csv'
FRESH_START = True
WEIGHT_COLUMN = ""
NUMBER_OF_CLASSES = 1
NUM_ROUND = 250
EARLY_STOPPING_ROUNDS = -1
VERBOSITY = 1
ETA = 0.1
GAMMA = 0.001
MAX_DEPTH = 10
MIN_CHILD_WEIGHT = 1
MAX_DELTA_STEP = 0
SUBSAMPLE = 1
COLSAMPLE_BYTREE = 1
COLSAMPLE_BYLEVEL = 1
COLSAMPLE_BYNODE = 1
REG_LAMBDA = 1
ALPHA = 0
SCALE_POS_WEIGHT = 1
OBJECTIVE = 'reg:squarederror'
TREE_METHOD = 'auto'
## AI Platform Training job related arguments:
REGION='us-central1'
SCALE_TIER='CUSTOM'
MASTER_MACHINE_TYPE='standard_gpu'
JOB_NAME="kaggle_xgboost_example"

get_ipython().system('gcloud ai-platform jobs submit training {JOB_NAME}     --master-image-uri gcr.io/aihub-content-external/kfp-components/trainer/dist_xgboost:latest     --region {REGION}     --scale-tier {SCALE_TIER}     --master-machine-type {MASTER_MACHINE_TYPE}     --     --training-data {TRAINING_DATA}     --target-column {TARGET_COLUMN}     --validation-data {VALIDATION_DATA}     --output-location {OUTPUT_LOCATION}     --data-type {DATA_TYPE}     --fresh-start {FRESH_START}     --weight-column {WEIGHT_COLUMN}     --number-of-classes {NUMBER_OF_CLASSES}     --num-round {NUM_ROUND}     --early-stopping-rounds {EARLY_STOPPING_ROUNDS}     --verbosity {VERBOSITY}     --eta {ETA}     --gamma {GAMMA}     --max-depth {MAX_DEPTH}     --min-child-weight {MIN_CHILD_WEIGHT}     --max-delta-step {MAX_DELTA_STEP}     --subsample {SUBSAMPLE}     --colsample-bytree {COLSAMPLE_BYTREE}     --colsample-bylevel {COLSAMPLE_BYLEVEL}     --colsample-bynode {COLSAMPLE_BYNODE}     --reg-lambda {REG_LAMBDA}     --alpha {ALPHA}     --scale-pos-weight {SCALE_POS_WEIGHT}     --objective {OBJECTIVE}     --tree-method {TREE_METHOD} ')


# # Download the trained XGBoost model
# 
# After the training run succeeds a model file will be exported to the GCS bucket defined by the `OUTPUT_LOCATION` parameter. Download this model file to generate inferences locally.

# In[ ]:


# Copy the trained model file locally
tf.io.gfile.copy(
    os.path.join(OUTPUT_LOCATION, 'model.bst'),
    'model.bst',
    overwrite=True)


# # Generate inferences for the test set
# 
# Generate inferences for the test dataset and write the predictions into a submission CSV file for Kaggle.

# In[ ]:


# Load the trained XGBoost model
bst = xgb.Booster({'nthread': 4})
bst.load_model('model.bst')

# Generate inferences for the test set
del test['SalePrice']
dmatrix = xgb.DMatrix(test)
test_pred = bst.predict(dmatrix)

# Write the test set inferences to a CSV file for submission
test_pred = pd.concat((test_ids, pd.DataFrame(test_pred)), axis=1)
test_pred.columns = ['Id', 'SalePrice']
test_pred.to_csv('test_predictions.csv', index=False)


# # Inspect the training job
# 
# After the training job completes on AI Platform Training, a run report will be created in the `OUTPUT_LOCATION`. The report examines the quality of the trained model and provides a visual inspection of the training and validation error from the training run.

# In[ ]:


tf.io.gfile.copy(
    os.path.join(OUTPUT_LOCATION, 'report.html'),
    'report.html',
    overwrite=True)

with open('report.html', 'r') as f:
    html_report = f.read()

display(HTML(html_report))


# In[ ]:




