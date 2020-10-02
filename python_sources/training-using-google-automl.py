#!/usr/bin/env python
# coding: utf-8

# # Set Parameters
# 
# In GCP, work is organized into projects.
# 
# **gcs_path** is the path that images and the training csv will be uploaded to. AutoML Vision requires the images to be stored in a directory with the root directory being gs://{project_id}-vcm. 
# 
# **train_filename** is the name of the csv file that's uploaded to GCS. Doesn't matter what you choose here. 
# 
# **gcp_service_account_json** is the path to the service account key. Service accounts allow you to authenticate with GCP using a JSON key (rather than typing in a password). I uploaded my service account key as a private dataset. Read more about setting one up at [https://cloud.google.com/iam/docs/understanding-service-accounts](https://cloud.google.com/iam/docs/understanding-service-accounts)
# 
# **train_budget** is the number of node hours. 1 is min. 24 is maximum. I believe AutoML Vision Classifaction costs $20 per node hour.
# 
# **dataset_name** is the name of the dataset that's loaded into AutoML. Doesn't matter what you choose here. 
# 
# **model_name** is the name of the model in AutoML. Doesn't matter what you choose here. I use the convention of dataset_name *underscore* train_budget
# 

# In[ ]:


gcp_project_id = 'kaggle-playground-170215'
gcs_path = "gs://{}-vcm/recursion-cellular-image-classification/RGB224/".format(gcp_project_id)
train_filename = "automl_train.csv"
gcp_service_account_json = '/kaggle/input/gcloudserviceaccountkey/kaggle-playground-170215-4ece6a076f22.json'
gcp_compute_region = 'us-central1' #for now, AutoML is only available in this region
train_budget = 24
dataset_name = 'recursion_224px_wo_controls'
model_name = "{}_{}".format(dataset_name,train_budget) 


# # Install Google Cloud SDK and AutoML Package
# Followed the instructions at [https://cloud.google.com/sdk/install](https://cloud.google.com/sdk/install) with some slight modifications for this environment. Need the Google Cloud SDK to use gsutil, which is the fast way to transfer the training data to Google Cloud Storage (GCS). 
# 
# Also need to install the AutoML Python Package. 

# In[ ]:


#google cloud SDK
get_ipython().system('echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list')
get_ipython().system('apt-get install apt-transport-https ca-certificates')
get_ipython().system('curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -')
get_ipython().system('apt-get update && apt-get install --yes --allow-unauthenticated google-cloud-sdk')

#AutoML package
get_ipython().system('pip install google-cloud-automl')


# In[ ]:


#authenticate the Google Cloud SDK
get_ipython().system('gcloud config set project $gcp_project_id')
get_ipython().system('gcloud auth activate-service-account --key-file=$gcp_service_account_json')

#uncomment if you don't already have this gcs bucket setup
#!gsutil mb -p $gcp_project_id -c regional -l $gcp_compute_region gs://$gcp_project_id-vcm/


# # Import libraries
# Importing libraries after all packages have been installed

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile
import os
from google.cloud import automl_v1beta1 as automl


# # Create the training CSV and upload to Google Cloud Storage (GCS)
# Here are the docs for what your data should look like for AutoML vision

# ## Split data by experiment
# By default, splitting betweeen train, validation and test is optional for AutoML (by default it splits randomly). But it's important for this use case. 

# In[ ]:


df_train = pd.read_csv('../input/recursion-2019-load-resize-and-save-images/new_train.csv')

validation = ['HEPG2-07','HUVEC-16','RPE-07','U2OS-02']
test = ['HEPG2-03','HUVEC-04','RPE-05','U2OS-01']


df_train['split'] = 'TRAIN' 
#put experiments in the lists aboe in Validation and TEST
df_train.loc[df_train['experiment'].isin(validation),'split'] = 'VALIDATION' 
df_train.loc[df_train['experiment'].isin(test),'split'] = 'TEST'


# ## Upload training CSV to GCS

# In[ ]:



#add gcs path
df_train['gcspath'] = gcs_path + 'train/' + df_train['filename']

#AutoML requires the label to be an int not a float
df_train['sirna'] = df_train['sirna'].astype(int)

df_train[['split','gcspath','sirna']].to_csv(train_filename,index=False,header=False)
get_ipython().system('gsutil cp $train_filename $gcs_path$train_filename #upload csv file to GCS')


# # Extract images and upload to GCS

# In[ ]:


with zipfile.ZipFile('../input/recursion-2019-load-resize-and-save-images/train.zip', 'r') as zip_ref:
    zip_ref.extractall('./train/')


# In[ ]:


get_ipython().system('gsutil -q -m cp -r ./train/* $gcs_path/ #upload images to gcs')
get_ipython().system('rm -r ./train/ #need to do this because otherwise you get a "too many output files error"')


# # Kick off an AutoML training job
# 
# Requires a three step process:
# 1. Setup an AutoML data object
# 2. Load data into the object
# 3. Train the model
# 
# This is mostly boilerplate copied from:
# [https://cloud.google.com/vision/automl/docs/tutorial](http://https://cloud.google.com/vision/automl/docs/tutorial)

# In[ ]:


#1. Setup AutoML Data Object
client = automl.AutoMlClient.from_service_account_json(gcp_service_account_json)
project_location = client.location_path(gcp_project_id, gcp_compute_region)

my_dataset = {
    "display_name": dataset_name,
    "image_classification_dataset_metadata": {"classification_type": "MULTICLASS"},
}

# Create a dataset with the dataset metadata in the region
dataset = client.create_dataset(project_location, my_dataset)
dataset_id = (dataset.name.split("/")[-1])


# In[ ]:


#2 Load data into the object
dataset_full_id = client.dataset_path(
    gcp_project_id, gcp_compute_region, dataset_id
)

input_uris = ('{}{}'.format(gcs_path ,train_filename)).split(",")
input_config = {"gcs_source": {"input_uris": input_uris}}

response = client.import_data(dataset_full_id, input_config)

print("Processing import...")
print("Data imported. {}".format(response.result()))


# In[ ]:


#3. Train the model
my_model = {
    "display_name": model_name,
    "dataset_id": dataset_id,
    "image_classification_model_metadata": {"train_budget": train_budget}
    if train_budget
    else {},
}

response = client.create_model(project_location, my_model)

print("Training operation name: {}".format(response.operation.name))
print("Training started...")

