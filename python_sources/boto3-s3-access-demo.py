#!/usr/bin/env python
# coding: utf-8

# <img width="200" align="left" src="https://www.ehu.eus/documents/1294053/6902569/Escuela+Ingenieria_Gipuzkoa_ingles_positivo_baja.jpg">

# # Boto3 S3 access demo
# It is interesting to try and get access to any of the services from AWS. In this case, we use boto3 to store some local files into an S3 bucket. The requirement to run this demo is to have credentials of an AWS account with S3 Access. We usually would store these credentials in a local file, but we have tried the Kaggle Secrets Add-on to see if it works.
# 
# If you fork this notebook, you will have to provide your keys in the Kaggle Secrets Add-on menu. Of course, you will have to toggle on the Internet option in the Settings menu to access AWS. We use a simple dataset named "txisteak", but you could try anything you want.
# 
# More info at this links:
# 
# AWS S3 storage service: https://aws.amazon.com/s3/
# 
# AWS boto3 library: https://aws.amazon.com/sdk-for-python/
# 
# Feature launch: User secrets | Kaggle: https://www.kaggle.com/product-feedback/114053
# 
# "Txisteak" dataset: https://www.kaggle.com/aitzolezeizaramos/txisteak/
# 

# In[ ]:


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
aws_id = user_secrets.get_secret("aws_access_key_id")
aws_key = user_secrets.get_secret("aws_secret_access_key")
aws_region = user_secrets.get_secret("aws_region")

#If the Secrets Add-on works properly, we will see the current region here
# By the way, we do not need a region, because S3 allows us to create Global buckets too

aws_region


# In[ ]:


import boto3
import uuid
import os

s3 = boto3.resource(
    's3',
    aws_access_key_id=aws_id,
    aws_secret_access_key=aws_key,
)

s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_id,
    aws_secret_access_key=aws_key,
)
bucket_name = ''.join(['kaggle2020', str(uuid.uuid4())])
bucket_response = s3_client.create_bucket(Bucket=bucket_name)

response = s3_client.list_buckets()

# Output the bucket names
#print('Existing buckets:')
#for bucket in response['Buckets']:
#    print(f'  {bucket["Name"]}')


for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
#        print(os.path.join(dirname, filename))
        file_name = os.path.join(dirname, filename)
        response = s3_client.upload_file(file_name, bucket_name, filename)

# Let us get some feedback: the list of the objects of our Bucket is very verbose and we can
# check that everything is OK

s3_client.list_objects_v2(Bucket=bucket_name)


# After running the notebook, the S3 bucket will look like this:

# ![S3%20bucket%20contents.png](attachment:S3%20bucket%20contents.png)

# In[ ]:


# Finally, we will delete the bucket to let everything as it was before

bucket = s3.Bucket(bucket_name)
# suggested by Jordon Philips 
bucket.objects.all().delete()
s3_client.delete_bucket(Bucket=bucket_name)

