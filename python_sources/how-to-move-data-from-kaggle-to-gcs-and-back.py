#!/usr/bin/env python
# coding: utf-8

# # How to move data from Kaggle to GCS and back
# * The purpose of this notebook is to demonstrate how to move data from Kaggle into the Google Storage Client
# * We also demonstrate how to move data from the Google Storage Client into a Kaggle notebook
# * Note that this requires enabling Google Cloud Services in the Add-ons menu of the notebook editor.

# **Step 1:** Import Python Modules

# In[ ]:


import os 
import pandas as pd
import pandas_profiling as pp


# **Step 2:** Import Data from Kaggle
# * Note that you add the data to your kernel by pressing the "add data" button in the top right corner of the notebook editor

# In[ ]:


wisconsin = '/kaggle/input/breast-cancer-wisconsin-data/data.csv'


# **Step 3:** Connect to GCS Storage Client and Define Some Helper Functions
# * https://cloud.google.com/storage/docs/

# In[ ]:


from google.cloud import storage
storage_client = storage.Client(project='kaggle-playground-170215')

def create_bucket(dataset_name):
    """Creates a new bucket. https://cloud.google.com/storage/docs/ """
    bucket = storage_client.create_bucket(dataset_name)
    print('Bucket {} created'.format(bucket.name))

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
    
def list_blobs(bucket_name):
    """Lists all the blobs in the bucket. https://cloud.google.com/storage/docs/"""
    blobs = storage_client.list_blobs(bucket_name)
    for blob in blobs:
        print(blob.name)
        
def download_to_kaggle(bucket_name,destination_directory,file_name):
    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""
    os.makedirs(destination_directory, exist_ok = True)
    full_file_path = os.path.join(destination_directory, file_name)
    blobs = storage_client.list_blobs(bucket_name)
    for blob in blobs:
        blob.download_to_filename(full_file_path)


# **Step 3:** Create a new GCS Bucket

# In[ ]:


bucket_name = 'wisconsinbreastcancer_test'         
try:
    create_bucket(bucket_name)   
except:
    pass


# **Step 3:** Upload your data to a GCS Bucket

# In[ ]:


local_data = '/kaggle/input/breast-cancer-wisconsin-data/data.csv'
file_name = 'data.csv' 
upload_blob(bucket_name, local_data, file_name)
print('Data inside of',bucket_name,':')
list_blobs(bucket_name)


# **Step 4:** Download your data from the GCS Bucket

# In[ ]:


destination_directory = '/kaggle/working/breastcancerwisconsin/'       
file_name = 'data.csv'
download_to_kaggle(bucket_name,destination_directory,file_name)


# **Step 5:** Preview the data that you just downloaded

# In[ ]:


os.listdir('/kaggle/working/breastcancerwisconsin/')


# In[ ]:


full_file_path = os.path.join(destination_directory, file_name)
new_file = pd.read_csv(full_file_path)
pp.ProfileReport(new_file)

