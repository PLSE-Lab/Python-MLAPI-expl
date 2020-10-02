#!/usr/bin/env python
# coding: utf-8

# # AutoML Tables Tutorial Notebook
# 
# Welcome to this step-by-step tutorial that will show you how to use Kaggle's new integration with [Google AutoML Tables](https://cloud.google.com/automl-tables/). 
# 
# Tables is a powerful AutoML Tool (AMLT) to handle structured data problems like regression and classification, and in this tutorial we will apply it to one of our favorite Kaggle Competitions: [Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). In this competition we try to use data about a house (square footage, location, etc) to predict it's Sale Price. 
# 
# Using AutoML Tables, we see we're able to get an initial high performing solution without spending upfront time on feature engineering, model selection or hyperparameter tuning. We hope you get a chance to try this tool and leave us your feedback!

# ## Important Notes about AutoML
# 
# Before we get started using AutoML, there's a few important things to know. 
# 
# #### Firstly, please note that Google AutoML is a paid service and requires a GCP project with billing enabled to use. Prices vary by the specific AutoML product you'd like to use; in this tutorial we use AutoML tables, which at the time of publishing costs \$19.32 per hour of compute during training and $1.16 per hour of compute for batch prediction with more details [here](https://cloud.google.com/automl-tables/pricing).
# 
# Furthermore, keep in mind that AutoML is currently in [Beta](https://cloud.google.com/products/#product-launch-stages). While we're excited to integrate with an early exciting technology, you may run into some usability frictions or known issues. We welcome all feedback from the community, and user feedback will help us improve the documentation and be shared with the AutoML team to improve the product.
# 
# Lastly, AutoML tends to work better on **larger datasets** and when trained for **longer amounts of time**. In this particular example, we pick a very small dataset (few thousand rows) where we  train a reasonable AutoML model within one to two hours. So, we picked this particular reference because it is an easy & familiar one for our Kaggle community to see how to use AutoML even though the tool would usually perform better on larger and/or more intensive problems.

# ## Step 0: Set up your Google Cloud Project
# 
# Before getting started with our AutoML Integration, you'll need to link a Google Account and create a GCP Project. You can see more details on how to do this in the official AutoML [docs](https://cloud.google.com/automl-tables/docs/quickstart). In particular, you'll want to:
# 
# 1. [Select or create a GCP Project](https://console.cloud.google.com/projectselector2/home/dashboard?_ga=2.199933678.-119742050.1571893794)
# 2. [Enable Billing](https://cloud.google.com/billing/docs/how-to/modify-project). Remember to understand pricing prior to this step.
# 3. [Enable Cloud AutoML and Storage APIs](https://console.cloud.google.com/flows/enableapi?apiid=storage-component.googleapis.com,automl.googleapis.com,storage-api.googleapis.com&redirect=https://console.cloud.google.com&_ga=2.32030718.-119742050.1571893794)
# 
# Once you have your account and project configured, simply attach it to this Kernel by selecting **Google Cloud Services** from the Add-ons menu in the Kernel editor and following the on-screen instructions to link your account. Make sure you've linked both AutoML and GCS for this tutorial. 
# 
# Luckily, you're eligible for a \$300 credit when creating a new GCP project, and we'll be looking to distribute more credits throughout the year to make it easier to try --- stay tuned!

# ## Step 1: Read in and Prepare your Data
# 
# AutoML reads data from internet enabled storage like Google Cloud Storage or BigQuery, so we need to move our Kaggle Dataset from our 'local' environment in our Kernel to a GCS bucket specifically configured for our use case. The next few code cells will walk through how to prepare the data once you've added the Housing Prices Competition Dataset to your Notebook. 
# 
# First read in the data into pandas dataframes, and do any cleaning & pre-processing you'd like to make your dataset easier to use. In this example, we make minimal changes to the data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")


# In[ ]:


# This dataset has some missing values, which we set to the median of the column for the purpose of this tutorial. 
cleaned_train_df = train_df.fillna(train_df.median())
cleaned_test_df = test_df.fillna(test_df.median())


# In[ ]:


# Any results you write to the current directory are saved as output.
# Write the dataframes back out to a csv file, which we can more easily upload to GCS. 
cleaned_train_df.to_csv(path_or_buf='train.csv', index=False)
cleaned_test_df.to_csv(path_or_buf='test.csv', index=False)


# ## Step 2: Initialize the clients and move your data to GCS
# 
# Now we want to take the csv files we prepared locally and upload them to a bucket in Google Cloud Storage that AutoML Tables can access. 
# 
# First, we'll create variables to store our GCS Project ID and come up with a new name for the GCS Bucket where this data will live. The `PROJECT_ID` should match the billing-enabled project you've set up using your linked Google account. We'll create a new GCS bucket and give it a `BUCKET_NAME` according to the GCS naming [guidelines](https://cloud.google.com/storage/docs/naming).
# 
# **Note: GCS Bucket names need to be globally unique and must contain only lowercase letters. For AutoML to work, the GCS Bucket must also exist in the `us-central1` region.**

# In[ ]:


#REPLACE THIS WITH YOUR OWN GOOGLE PROJECT ID
PROJECT_ID = 'kaggle-playground-170215'
#REPLACE THIS WITH A NEW BUCKET NAME. NOTE: BUCKET NAMES MUST BE GLOBALLY UNIQUE
BUCKET_NAME = 'automl-tutorial'
#Note: the bucket_region must be us-central1.
BUCKET_REGION = 'us-central1'


# From there, we'll use our account with the AutoML and GCS libraries to initialize the clients we can use to do the rest of our work. The code below is boilerplate you can use directly, assuming you've entered your own `PROJECT_ID` and `BUCKET_NAME` in the previous step.

# In[ ]:


from google.cloud import storage, automl_v1beta1 as automl

storage_client = storage.Client(project=PROJECT_ID)
tables_gcs_client = automl.GcsClient(client=storage_client, bucket_name=BUCKET_NAME)
automl_client = automl.AutoMlClient()
# Note: AutoML Tables currently is only eligible for region us-central1. 
prediction_client = automl.PredictionServiceClient()
# Note: This line runs unsuccessfully without each one of these parameters
tables_client = automl.TablesClient(project=PROJECT_ID, region=BUCKET_REGION, client=automl_client, gcs_client=tables_gcs_client, prediction_client=prediction_client)


# In[ ]:


# Create your GCS Bucket with your specified name and region (if it doesn't already exist)
bucket = storage.Bucket(storage_client, name=BUCKET_NAME)
if not bucket.exists():
    bucket.create(location=BUCKET_REGION)


# In order to actually move my local files to GCS, I've copied over a few helper functions from another helpful tutorial [Notebook](https://www.kaggle.com/paultimothymooney/demo-moving-data-to-and-from-gcs-and-kaggle) on moving Kaggle data to GCS. 

# In[ ]:


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
    
def download_to_kaggle(bucket_name,destination_directory,file_name,prefix=None):
    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""
    os.makedirs(destination_directory, exist_ok = True)
    full_file_path = os.path.join(destination_directory, file_name)
    blobs = storage_client.list_blobs(bucket_name,prefix=prefix)
    for blob in blobs:
        blob.download_to_filename(full_file_path)


# Now I just run those functions on my `train.csv` and `test.csv` files saved locally and all my data is in the right place within Google Cloud Storage.

# In[ ]:


upload_blob(BUCKET_NAME, 'train.csv', 'train.csv')
upload_blob(BUCKET_NAME, 'test.csv', 'test.csv')


# ## Step 3: Train an AutoML Model
# 
# I'll break down the training step for AutoML into three operations:
# 
# 1. Importing the data from your GCS bucket to your autoML client
# 2. Specifying the target you want to predict on your dataset
# 3. Creating your model
# 

# #### Importing from GCS to AutoML
# 
# The first step is to create a dataset within AutoML tables that references your saved data in GCS. This is relatively straight forward, first just simply choose a name for your dataset.

# In[ ]:


dataset_display_name = 'housing_prices'
new_dataset = False
try:
    dataset = tables_client.get_dataset(dataset_display_name=dataset_display_name)
except:
    new_dataset = True
    dataset = tables_client.create_dataset(dataset_display_name)


# And next, give it the path to where the relevant data is in GCS (GCS file paths follow the format `gs://BUCKET_NAME/file_path`) and import your data. 

# In[ ]:


# gcs_input_uris have the familiar path of gs://BUCKETNAME//file

if new_dataset:
    gcs_input_uris = ['gs://' + BUCKET_NAME + '/train.csv']

    import_data_operation = tables_client.import_data(
        dataset=dataset,
        gcs_input_uris=gcs_input_uris
    )
    print('Dataset import operation: {}'.format(import_data_operation))

    # Synchronous check of operation status. Wait until import is done.
    import_data_operation.result()
# print(dataset)


# #### Select the Target in your dataset
# 
# Now specify which column in your dataset is the target, and which is an ID column (if any). In our case, the `TARGET_COLUMN` is *SalePrice*.

# In[ ]:


model_display_name = 'tutorial_model'
TARGET_COLUMN = 'SalePrice'
ID_COLUMN = 'Id'

# TODO: File bug: if you run this right after the last step, when data import isn't complete, you get a list index out of range
# There might be a more general issue, if you provide invalid display names, etc.

tables_client.set_target_column(
    dataset=dataset,
    column_spec_display_name=TARGET_COLUMN
)


# In[ ]:


# Make all columns nullable (except the Target and ID Column)
for col in tables_client.list_column_specs(PROJECT_ID,BUCKET_REGION,dataset.name):
    if TARGET_COLUMN in col.display_name or ID_COLUMN in col.display_name:
        continue
    tables_client.update_column_spec(PROJECT_ID,
                                     BUCKET_REGION,
                                     dataset.name,
                                     column_spec_display_name=col.display_name,
                                     type_code=col.data_type.type_code,
                                     nullable=True)


# #### Create your model
# 
# Now the moment you've all been waiting for, the actual training step! Run the following code below to actually train an AutoML model using the setup we've described. 
# In TRAIN_BUDGET, you can set the **maximum** amount of time AutoML is allowed to run for, which helps manage both time and cost. Generally, the longer you allow AutoML to run, the better results you can expect and it will automatically stop early if it finds the optimal solution sooner than your allocated budget. 
# 
# **Note: `TRAIN_BUDGET` is specified in milli-hours, so `1000` refers to 1 hour of wall clock time. Spending more time training (setting a higher `TRAIN_BUDGET`) is expected to result in more accurate models.**

# In[ ]:


# Train the model. This will take hours (up to your budget). AutoML will early stop if it finds an optimal solution before your budget.
# On this dataset, AutoML usually stops around 2000 milli-hours (2 hours)

TRAIN_BUDGET = 2000 # (specified in milli-hours, from 1000-72000)
model = None
try:
    model = tables_client.get_model(model_display_name=model_display_name)
except:
    response = tables_client.create_model(
        model_display_name,
        dataset=dataset,
        train_budget_milli_node_hours=TRAIN_BUDGET,
        exclude_column_spec_names=[ID_COLUMN, TARGET_COLUMN]
    )
    print('Create model operation: {}'.format(response.operation))
    # Wait until model training is done.
    model = response.result()
# print(model)


# ## Step 4: Batch Predict on your Test Dataset 
# 
# Now we're ready to see what AutoML can do! We'll use some code that should look familiar to point our newly created autoML model to our test file and spit out some new predictions. 
# 
# Go ahead and select the `gcs_input_uris` based on the location of where your test data is within GCS, and choose a `gcs_output_uri_prefix` that relates to the path where you'd like your predictions written out to once done.

# In[ ]:


gcs_input_uris = 'gs://' + BUCKET_NAME + '/test.csv'
gcs_output_uri_prefix = 'gs://' + BUCKET_NAME + '/predictions'

batch_predict_response = tables_client.batch_predict(
    model=model, 
    gcs_input_uris=gcs_input_uris,
    gcs_output_uri_prefix=gcs_output_uri_prefix,
)
print('Batch prediction operation: {}'.format(batch_predict_response.operation))
# Wait until batch prediction is done.
batch_predict_result = batch_predict_response.result()
batch_predict_response.metadata


# ## Step 5: Download your predictions
# 
# Congratulations on successfully running batch prediction! Your results can be found within your GCS bucket, and we can use our helper functions from before to download them from GCS into an environment we can work with more easily within our Notebook.

# In[ ]:


# The output directory for the prediction results exists under the response metadata for the batch_predict operation
# Specifically, under metadata --> batch_predict_details --> output_info --> gcs_output_directory
# Then, you can remove the first part of the output path that contains the GCS Bucket information to get your desired directory
gcs_output_folder = batch_predict_response.metadata.batch_predict_details.output_info.gcs_output_directory.replace('gs://' + BUCKET_NAME + '/','')
download_to_kaggle(BUCKET_NAME,'/kaggle/working','tables_1.csv', prefix=gcs_output_folder)


# From here, you're pretty much done! In the last piece of code below, we'll simply generate a csv file in the format that we can use to submit to the competition: namely, with two columns only for `ID` and the predicted `SalePrice`. 

# In[ ]:


preds_df = pd.read_csv("tables_1.csv")
submission_df = preds_df[['Id', 'predicted_SalePrice']]
submission_df.columns = ['Id', 'SalePrice']
submission_df.to_csv('submission.csv', index=False)


# ## Done! 
# 
# Congratulations on using AutoML Tables to solve a Kaggle Competition end-to-end! Please leave us your feedback and questions. 
