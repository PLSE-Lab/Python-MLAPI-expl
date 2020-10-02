#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from datetime import datetime

from sklearn.model_selection import train_test_split

from google.cloud import storage
from google.cloud import automl_v1beta1 as automl
from google.cloud import automl_v1beta1 as automl

from automlwrapper import AutoMLWrapper


# This notebook utilizes a utility script that wraps much of the AutoML Python client library, to make the code in this notebook easier to read. Feel free to check out the utility for all the details on how we are calling the underlying AutoML Client Library!

# In[ ]:


# Set your own values for these. bucket_name should be the project_id + '-lcm'.
PROJECT_ID = 'automl-kaggle-265409'
bucket_name = 'automl-kaggle-bucket'

region = 'us-central1' # Region must be us-central1
dataset_display_name = 'kaggle_tweets'
model_display_name = 'kaggle_starter_model1'

storage_client = storage.Client(project=PROJECT_ID)
client = automl.AutoMlClient()


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


nlp_train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
nlp_test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
def callback(operation_future):
    result = operation_future.result()


# In[ ]:


nlp_train_df.tail()


# > ### Data spelunking
# How often does 'fire' come up in this dataset?

# In[ ]:


nlp_train_df.loc[nlp_train_df['text'].str.contains('fire', na=False, case=False)]


# Does the presence of the word 'fire' help determine whether the tweets here are real or false?

# In[ ]:


nlp_train_df.loc[nlp_train_df['text'].str.contains('fire', na=False, case=False)].target.value_counts()


# ### GCS upload/download utilities
# These functions make upload and download of files from the kernel to Google Cloud Storage easier. This is needed for AutoML

# In[ ]:


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}'.format(
        source_file_name,
        'gs://' + bucket_name + '/' + destination_blob_name))
    
def download_to_kaggle(bucket_name,destination_directory,file_name,prefix=None):
    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""
    os.makedirs(destination_directory, exist_ok = True)
    full_file_path = os.path.join(destination_directory, file_name)
    blobs = storage_client.list_blobs(bucket_name,prefix=prefix)
    for blob in blobs:
        blob.download_to_filename(full_file_path)


# In[ ]:


bucket = storage.Bucket(storage_client, name=bucket_name)
if not bucket.exists():
    bucket.create(location=region)


# ### Export to CSV and upload to GCS

# In[ ]:


# Select the text body and the target value, for sending to AutoML NL
nlp_train_df[['text','target']].to_csv('train.csv', index=False, header=False) 


# In[ ]:


nlp_train_df[['id','text','target']].head()


# In[ ]:


training_gcs_path = 'uploads/kaggle_getstarted/full_train.csv'
upload_blob(bucket_name, 'train.csv', training_gcs_path)


# ## Create our class instance

# In[ ]:


amw = AutoMLWrapper(client=client, 
                    project_id=PROJECT_ID, 
                    bucket_name=bucket_name, 
                    region='us-central1', 
                    dataset_display_name=dataset_display_name, 
                    model_display_name=model_display_name)
       


# ## Create (or retreive) dataset
# Check to see if this dataset already exists. If not, create it

# In[ ]:


if not amw.get_dataset_by_display_name(dataset_display_name):
    print('dataset not found')
    amw.create_dataset()
    amw.import_gcs_data(training_gcs_path)

amw.dataset


# ## Kick off the training for the model
# And retrieve the training info after completion. 
# Start model deployment.

# In[ ]:


if not amw.get_model_by_display_name():
    amw.train_model()
amw.deploy_model()
amw.model


# In[ ]:


amw.model_full_path


# ## Prediction
# Note that prediction will not run until deployment finishes, which takes a bit of time.
# However, once you have your model deployed, this notebook won't re-train the model, thanks to the various safeguards put in place. Instead, it will take the existing (trained) model and make predictions and generate the submission file.

# In[ ]:


nlp_test_df.head()


# In[ ]:


# Create client for prediction service.
prediction_client = automl.PredictionServiceClient()
amw.set_prediction_client(prediction_client)


# In[ ]:


# nlp_test_df.iloc[:10][['text']].to_csv('test.csv', index=False, header=False)
# testing_gcs_path = 'uploads/kaggle_getstarted/test.csv'
# upload_blob(bucket_name, 'test.csv', testing_gcs_path)

# nlp_test_df[['text']].to_csv('test.csv', index=False, header=False) 
# pd.DataFrame({"files": ['gs://{}/{}'.format(bucket_name, testing_gcs_path)]}).to_csv('files.csv', index=False, header=False) 

# files_gcs_path = 'uploads/kaggle_getstarted/files.csv'
# upload_blob(bucket_name, 'files.csv', files_gcs_path)

# input_uri = 'gs://{}/{}'.format(bucket_name, files_gcs_path)
# output_uri = 'gs://{}/{}/'.format(bucket_name, 'uploads/kaggle_getstarted/test_results.csv')
# model_id = 'TCN6064180461138083840'
# # prediction_client = automl.PredictionServiceClient()
# # amw.set_prediction_client(prediction_client)

# print(input_uri)
# # Get the full path of the model.
# model_full_id = prediction_client.model_path(
#     PROJECT_ID, region, model_id)

# gcs_source = automl.types.GcsSource(
#     input_uris=[input_uri])

# input_config = automl.types.BatchPredictInputConfig(gcs_source=gcs_source)
# gcs_destination = automl.types.GcsDestination(
#     output_uri_prefix=output_uri)
# output_config = automl.types.BatchPredictOutputConfig(
#     gcs_destination=gcs_destination)
# # [0.0-1.0] Only produce results higher than this value
# params = {'score_threshold': '0.5'}

# timeout = 60 * 60 # 1 hour
# response = prediction_client.batch_predict(
#     model_full_id, input_config, output_config, params, timeout=timeout)

# print('Waiting for operation to complete...')
# print(u'Batch Prediction results saved to Cloud Storage bucket. {}'.format(
#     response.result()))


# In[ ]:


import time
import pandas as pd
from tqdm.notebook import trange, tqdm
import random

batch_predictions = []
limit = nlp_test_df.shape[0]
step = 50

for i in trange(0, limit, step):
    try:
        batch_predictions.append(amw.get_predictions(nlp_test_df.iloc[i:i+step], input_col_name='text', 
                     limit=None, threshold=0.5, verbose=False))
    except:
        print('Error in batch {}'.format(i//step))
        time.sleep(3 * 60)
    time.sleep(3 * 60)

predictions_df = pd.concat(batch_predictions).reset_index()


# In[ ]:


nlp_test_df.head()


# In[ ]:


predictions_df.head()


# In[ ]:


# predictions_df = amw.get_predictions(nlp_test_df, 
#                                      input_col_name='text', 
# #                                      ground_truth_col_name='target', # we don't have ground truth in our test set
#                                      limit=None, 
#                                      threshold=0.5,
#                                      verbose=False)


# ## (optional) Undeploy model
# Undeploy the model to stop charges

# In[ ]:


# amw.undeploy_model()


# ## Create submission output

# In[ ]:


predictions_df.head()


# In[ ]:


submission_df = nlp_test_df.join(predictions_df['class'])[['id', 'class']]
submission_df.head()


# In[ ]:


submission_df = submission_df.rename(columns={'class':'target'})
submission_df.head()


# ## Submit predictions to the competition!

# In[ ]:


submission_df.to_csv("submission.csv", index=False, header=True)


# In[ ]:


get_ipython().system(' ls -l submission.csv')

