#!/usr/bin/env python
# coding: utf-8

# # Hello AutoML!
# This notebook covers *Google AutoML Natural Language* usege through tweet data classification problem. You can taste AutoML service which might be next step of ML age. 
# 
# This notebook is my first notebook. I want to share my experience with you! :)

# ## What is *AutoML*?

# AutoML is the process of automating the end-to-end process of applying machine learning to real-world problems. AutoML makes machine learning available in a true sense, even to people with no major expertise in this field.
# https://heartbeat.fritz.ai/automl-the-next-wave-of-machine-learning-5494baac615f
# 
# But not only for beginner, it is good to AI expert also. Using AutoML, we are able to get an initial high performing solution without spending upfront time on feature engineering, model selection or hyperparameter tuning.    
# In fact, aren't we already use 'grid-search' or 'argparse'? AutoML might act the role more fancy and easy instead of us.

# **Google Cloud AutoML service** offers several AutoML products, like AutoML Vision, AutoML Translation, AutoML Natural Language, etc.
# https://cloud.google.com/automl
# 
# Among them, this notebook focus on **how to use the "Google Cloud AutoML Natural Language" service using a kaggle kernel step by step with commment about diverse errors you can encounter.**
# 
# Through the AutoML NL service, **you can train model fit to your data. But you have no influence on model structure or hyperparameter tuning.** So to make a difference, you should focuse on preprocessing.
# 
# One of the common confusions is to think "Google Cloud AutoML Natural Language" and "Google Natural Language API" service as the same thing. But they are different services.
# 
# ![gg](https://bs-uploads.toptal.io/blackfish-uploads/uploaded_file/file/25738/image-1562143062720-f7420426bbaa2da13ed582fbca314219.png)
# 
# If you want to know more details about Google AutoML, I strongly recommand this two docs.
# - [NLP With Google Cloud Natural Language API by MAXIMILIAN HOPF](https://www.toptal.com/machine-learning/google-nlp-tutorial)
# - [Google Cloud AutoML NL official docs](https://cloud.google.com/natural-language/automl/docs/how-to)
# 
# Almost every code referred to below kernel. Thanks to yufengg and devvret.
# - [AutoML Getting Started Notebook](https://www.kaggle.com/yufengg/automl-getting-started-notebook) 
# - [AutoML Tables Tutorial Notebook](https://www.kaggle.com/devvret/automl-tables-tutorial-notebook) 

# ## Important Notes about Google Cloud AutoML
# There's a few important basic information you need to know.
# 
# 1. Please note that Google AutoML is a **paid service**
#    
#    But dont worry. They give us $300 free credit when creating a new GCP project, and bonus credit when newly start the google AutoML NL service. Also they don't charge automatically if you don't agree it explicitly.
#     - [AutoML Pricing policy](https://cloud.google.com/natural-language/automl/pricing?_ga=2.154004159.-640232925.1583643870&_gac=1.207751462.1584364219.EAIaIQobChMI66KVk8ae6AIV2aqWCh3VQwCxEAAYASAAEgI7ZvD_BwE) 
#     - You can check your billing account status [here](https://console.cloud.google.com/billing)
# ![Imgur](https://i.imgur.com/gHLTZoq.png)
# 
# 2. AutoML tends to work better on larger datasets and **takes a lot of times for training**. In our example, we pick a small dataset (few thousand rows) from ["Real or Not? NLP with Disaster Tweets"](https://www.kaggle.com/c/nlp-getting-startedwhere) competition. Even for this small dataset, it takes about 3~4 hours for trining.
# 
# 3. Understanding of GCS structure would help you.
#     - One GCS account can have several projects. The project corresponds to one application. 
#     - A project has a storage you can upload or download data. The storage has serveral buckets.
#     - You can make serveral datasets in your project. From one dataset, one model is trained.
# 
# So, **we are going to create GCP project and upload our data to certain bucket. And then, we can make our dataset and develope model.**

# ## Step 0: Set up and link your Google Cloud Project
# 
# If you use AutoML on your local enviroments, you need to do some enviroment settings. But through Kaggle kernel, the setting is so easy. Kaggle kernel is a good choice for taste it!
# 
# Follow this 3steps.
# 1. [Select or create a GCP Project](https://console.cloud.google.com/projectselector2/home/dashboard?_ga=2.199933678.-119742050.1571893794)
# 2. [Enable Billing](https://cloud.google.com/billing/docs/how-to/modify-project)
# 
# Once you have your account and project configured, simply attach it to this Kernel by selecting **Google Cloud Services** from the Add-ons menu in the Kernel editor and following the on-screen instructions to link your account.
# 3. [Enable Cloud AutoML and Storage APIs](https://console.cloud.google.com/flows/enableapi?apiid=storage-component.googleapis.com,automl.googleapis.com,storage-api.googleapis.com&redirect=https://console.cloud.google.com&_ga=2.32030718.-119742050.1571893794)
# 
# ![Imgur](https://i.imgur.com/z34qB09.png)
# ![Imgur1](https://i.imgur.com/z7wQij6.png)
# Note1. You should enable the **all of three services**, google Bigquery, Cloud AutoML, Storage APIs services.   
# 
# 
# (You can see more details on how to do this in the official AutoML [docs](https://cloud.google.com/automl-tables/docs/quickstart).)
# 
# Now we are ready. Let's go!

# ## Step 1: Read in and Prepare your Data
# 
# First read in the data into pandas dataframes, and do any pre-processing you want to. In this example, we skip the pre-processing step.

# In[ ]:


# Python related library
import numpy as np
import pandas as pd
import time
from datetime import datetime


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Import original datasets
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# ------ Insert your pre-processing code ---------
# 

# ## Step 2: Initialize the clients and move your data to GCS
# 
# At the step 1, we created GCP project. So we are going 
# 
# AutoML reads data from internet enabled storage like Google Cloud Storage that AutoML can access. So at this step, we are going to upload dataset from our 'local' environment in our Kernel to a GCS bucket. 
# 

# ### Automl project setting
# 
# We'll create variables to store our GCS Project ID(made in "Step 0") and come up with a new name for the GCS Bucket where this data will live. 
# - The `PROJECT_ID` should match the billing-enabled project you've set up using your linked Google account. 
# 
# - The `BUCKET_NAME` should be according to the GCS naming [guidelines](https://cloud.google.com/storage/docs/naming).   
# *Note: GCS Bucket names need to be globally unique and must contain only lowercase letters, numbers, dashes (-), underscores (_), and dots (.). Spaces are not allowed. Names containing dots require verification.*

# In[ ]:


# google.cloud related library
from google.cloud import storage, automl_v1beta1 as automl

# workaround to fix gapic_v1 error
from google.api_core.gapic_v1.client_info import ClientInfo

from automlwrapper import AutoMLWrapper


# In[ ]:


#####-- REPLACE the 'PROJECT_ID' and 'BUCKET_NAME with YOUR OWN --#####
PROJECT_ID = 'kaggle-nlp-wdt'
#NOTE: BUCKET NAMES MUST BE GLOBALLY UNIQUE
# and contain only lowercase letters, numbers, dashes (-), underscores (_), and dots (.). 
BUCKET_NAME = 'kaggle-nlp-wdt-lcm'


#Don't change this. AutoML currently is only eligible for region us-central1. So region must be us-central1. 
region = 'us-central1'

storage_client = storage.Client(project=PROJECT_ID)

# adding ClientInfo here to get the gapic_v1 call in place
client = automl.AutoMlClient(client_info=ClientInfo())

print(f'Starting AutoML notebook at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')


# Sometimes above code give `DefaultCredentialsError`.   
# First, you shoud confirm whether follow the *step 0-3.Enable Cloud AutoML and Storage APIs*. And click [Run]-[Restart session] menu.
# 
# Note: You should do *'restart session'*. It is not same with "refresh the browser page"

# In[ ]:


#####-- REPLACE this with YOUR OWN --#####
VERSION = 'V19'
BUCKET_PATH = 'preprocessing/'+VERSION+'/'
FILE_NAME = 'train_cleaned'+'_'+VERSION

# REPLACE THIS as you want. 
# Note: The dataset_display_name and model_display_name should contain only letters, numbers and underscores.
training_gcs_path = BUCKET_PATH+FILE_NAME+'.csv'
dataset_display_name = FILE_NAME 
model_display_name = 'model_'+FILE_NAME


# ### Dataset manipulation
# The structure of our data should meet the syntax guidlines of Google Cloud AutoML NL.   
# 
# You can use not only csv files, but also pdf.   
# *The CSV file has rows for each training document(content) and label column like `document content [,label]`, without index and column names.* The content column can be quoted in-line text or a Cloud Storage URI.
# 
# For more information, check the docs: https://cloud.google.com/natural-language/automl/docs/prepare

# In[ ]:


train.shape


# In[ ]:


# Select only the text body and the target value, for sending to AutoML NL
# Notice the options, "index=False, header=False"

train.loc[:,['text','target']].drop_duplicates()                                .to_csv('train.csv', index=False, header=False)


# ### Upload datasets to GCS bucket
# This step is just uploading our dataset from our 'local' environment to your GCS bucket. **Don't be confused with "Create dataset" work in step3.**

# In[ ]:


# Create your GCS Bucket with your specified name and region (if it doesn't already exist)
bucket = storage.Bucket(storage_client, name=BUCKET_NAME)
if not bucket.exists():
    bucket.create(location=BUCKET_REGION)


# In[ ]:


# These functions make upload and download of files from the kernel to Google Cloud Storage easier.
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


# If file has same path and name exists, file would be overwrited
upload_blob(BUCKET_NAME, 'train.csv', training_gcs_path)


# You can check the uploaded files at https://console.cloud.google.com/storage/browser

# ## Step 3: Construct Model

# ### Create our class instance

# In[ ]:


amw = AutoMLWrapper(client=client, 
                    project_id=PROJECT_ID, 
                    bucket_name=BUCKET_NAME, 
                    region='us-central1', 
                    dataset_display_name=dataset_display_name, 
                    model_display_name=model_display_name)
       


# ### Create (or retreive) dataset
# Check to see if this dataset already exists. If not, create it.
# 
# This step takes about 30 minute.

# In[ ]:


print(f'Getting dataset ready at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')
if not amw.get_dataset_by_display_name(dataset_display_name):
    print('dataset not found')
    amw.create_dataset()
    amw.import_gcs_data(training_gcs_path)

amw.dataset
print(f'Dataset ready at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')


# You can check the datasets you made at https://console.cloud.google.com/natural-language/datasets

# ## Step 4: Kick off the training for the model
# Using the dataset, we can construct out model. Then, we are going to deploy it. Deployment means *'switch on'* our model. 
# 
# This step could takes 3~4hours or more.

# In[ ]:


# Train the model. This will take hours (up to your budget). AutoML will early stop if it finds an optimal solution before your budget.

print(f'Getting model trained at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')

if not amw.get_model_by_display_name(model_display_name):
    print(f'Training model at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')
    amw.train_model()

print(f'Model trained. Ensuring model is deployed at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')
amw.deploy_model()
amw.model
print(f'Model trained and deployed at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')


# In[ ]:


amw.model_full_path


# ## Step 5: Prediction and submission
# Note that prediction will not run until deployment finishes, which takes a bit of time.
# However, once you have your model deployed, this notebook won't re-train the model, thanks to the various safeguards put in place. Instead, it will take the existing (trained) model and make predictions and generate the submission file.

# In[ ]:


test.head()


# In[ ]:


print(f'Begin getting predictions at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')

# Create client for prediction service.
prediction_client = automl.PredictionServiceClient()
amw.set_prediction_client(prediction_client)

predictions_df = amw.get_predictions(test, 
                                     input_col_name='text', 
#                                      ground_truth_col_name='target', # we don't have ground truth in our test set
                                     limit=None, 
                                     threshold=0.5,
                                     verbose=False)

print(f'Finished getting predictions at {datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S UTC")}')


# ### (optional) Undeploy model
# You can undeploy the model to stop charges
amw.undeploy_model()
# ### Submit predictions to the competition!

# In[ ]:


predictions_df.head()


# In[ ]:


submission_df = pd.concat([test['id'], predictions_df['class']], axis=1)


# In[ ]:


submission_df = submission_df.rename(columns={'class':'target'})
submission_df.head()


# In[ ]:


submission_df.to_csv("submission.csv", index=False, header=True)


# In[ ]:


get_ipython().system(' ls -l submission.csv')

