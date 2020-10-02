#!/usr/bin/env python
# coding: utf-8

# Kaggle kernels can be a bit of a pain to extract data from sometimes since you can't get the output until you commit the notebook, and if it got some neural network training in, it might take a _verrrrrrry_ long time... and in the meantime, if you leave your notebook for a few hours and it restarts, you lost all your hard work, so I've figure out how to transfer intermediate results to gcs:

# 1. Install google cloud storage python sdk

# In[ ]:


get_ipython().system('pip install --upgrade google-cloud-storage')


# 2.setup your auth keys as per https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python, download the authentication json file to your computer, then join all the split lines so it's one long string (the `cmd+J` shortcut if you are using atom is a useful option)

# save your gcs json into a json file on the kernel env by dumping it in through the jupyter widget text box as per below (the textbox destroys itself after running so you don't need to worry about having keys shown if you share your kernel)

# In[ ]:


from ipywidgets import interact, widgets
from IPython.display import display, clear_output
import json
import os
text = widgets.Text(
    value='my gcs.json auth',
    placeholder='Paste your gcs json auth file here!',
    description='Paste your gcs json auth file:',
    disabled=False
)
display(text)

def callback(text):
    # replace by something useful
    text = text.value.replace('\n', '\\n')
    
    try:
        json.loads(text)
        with open ("gcs.json", "w") as f:
            f.write(text) 
        
    except Exception as e:
        print(e)
    clear_output()
    

text.on_submit(callback)


# 3.Create your bucket for storing stuff

# In[ ]:


from google.cloud import storage

# Instantiates a client
if os.path.isfile('gcs.json') 
    storage_client = storage.Client.from_service_account_json(
            'gcs.json')
# -- uncomment the below to create a new bucket--
# The name for the new bucket
# bucket_name = 'my-bucket-name'

# # Creates the new bucket
# bucket = storage_client.create_bucket(bucket_name)

# print('Bucket {} created.'.format(bucket.name))


# 4.use the below to upload, e.g. 
# `upload_blob(storage_client, "my-bucket", "models/stage-2.pth", "stage-2.pth")`

# In[ ]:



def upload_blob(storage_client, bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))


# In[ ]:




