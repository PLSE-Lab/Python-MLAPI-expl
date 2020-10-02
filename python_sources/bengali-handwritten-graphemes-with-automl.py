#!/usr/bin/env python
# coding: utf-8

# # Classification of Bengali Handwritten Graphemes with Google AutoML

# This Notebook is intended as a starting point to use Google AutoML in image classification. Its creation is motivated by [this post](https://www.kaggle.com/c/bengaliai-cv19/discussion/122924).
# 
# Disclaimer: Googel AutoML is not for free and actually costs quit a bit if you train big models. As of January 2020, new users get a $300 credit to try the google cloud platform including AutoML

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


# In[ ]:


train_labels=pd.read_csv('../input/bengaliai-cv19/train.csv')
test_labels=pd.read_csv('../input/bengaliai-cv19/test.csv')
class_map=pd.read_csv('../input/bengaliai-cv19/class_map.csv')
sample_submission=pd.read_csv('../input/bengaliai-cv19/sample_submission.csv')


# In[ ]:


class_map.head()


# In[ ]:


train_labels.head()


# ## Setting up the Google cloud

# For settingup the google.cloud.storage class look [here](https://cloud.google.com/storage/docs/reference/libraries#client-libraries-install-python) and for more details consult the [documentation](https://googleapis.dev/python/storage/latest/client.html). Note that in google cloud storage everything is in stored in buckets. Unlike folders in an os they are not intended to be stacked.
# 
# Documentation for google AutoMl can be found [here](https://googleapis.dev/python/automl/latest/index.html) and a detailed Tutorial (for AutoMLVision) in multiple Programming Languages can be found [here](https://cloud.google.com/vision/automl/docs/tutorial). See also [this notebook](https://www.kaggle.com/devvret/automl-tables-tutorial-notebook) containing a AutoML tabels tutorial.

# In[ ]:


# Set your own project id here
PROJECT_ID = 'noble-return-265322'
BUCKET_REGION = 'us-central1'#europe-west3 is frankfurt but other 
                             #regions than the first are not supp'd  

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)

# The name for the new bucket
BUCKET_NAME = 'bengaliai'

# Creates the new bucket
bucket=storage.Bucket(storage_client,name=BUCKET_NAME)
if not bucket.exists():
    bucket.create(location=BUCKET_REGION)

print("Bucket {} created.".format(BUCKET_NAME))


# In[ ]:


#to upload data (datapoint=blob) to a bucket

def upload_blob(bucket_name, source_file_name, destination_blob_name,printyes=False):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"
    
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    
    if printyes:
        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
                )
            )
        
def download_to_kaggle(bucket_name,destination_directory,file_name,prefix=None):
    """
    Takes the data from your GCS Bucket and puts it
    into the working directory of your Kaggle notebook
    """
    os.makedirs(destination_directory, exist_ok = True)
    full_file_path = os.path.join(destination_directory, file_name)
    blobs = storage_client.list_blobs(bucket_name,prefix=prefix)
    for blob in blobs:
        blob.download_to_filename(full_file_path)


# We are now ready to import our data into the bucket. For that we first transform the data (i.e. the graphemes) into actual images. The data is given as rows in a csv each row being the concatenates pixels of a 137x236 grayscale images.

# In[ ]:


HEIGHT = 137
WIDTH = 236
N_CHANNELS=1


# In[ ]:


i=0 
name=f'train_image_data_{i}.parquet'
train_img = pd.read_parquet('../input/bengaliai-cv19/'+name)


# In[ ]:


train_img.shape


# In[ ]:


train_img.head()


# In[ ]:


# Visualize few samples of current training dataset
from matplotlib import pyplot as plt

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))
count=0
for row in ax:
    for col in row:
        col.imshow(train_img.iloc[[count]].drop(['image_id'],axis=1).to_numpy(dtype=np.float32).reshape(HEIGHT, WIDTH).astype(np.float64),cmap='binary')
        count += 1
plt.show()


# We use the 'image_id' attribute as the URI for our cloud storage bucket and copy the images (saved as PNG).
# 
# The follwoing code does this for one image in the dataset. If this was succesfull you can also view the file in the google [cloud storage browser](https://console.cloud.google.com/storage/browser).

# In[ ]:


from matplotlib.image import imsave
img=train_img.iloc[1]
image_path=img.image_id+'.png'
imsave(image_path,img.drop(['image_id']).to_numpy(dtype=np.float64).reshape(HEIGHT, WIDTH).astype(np.float64))

upload_blob(BUCKET_NAME,image_path,image_path,printyes=True)


# We do this now for all the images in the training set and afterwards also for the images in the test set. In order not to clutter our workingdirectory we delete every image after completing the upload.

# In[ ]:


from matplotlib.image import imsave
from os import remove

def upload_bengaliai_data(path):
    train_img = pd.read_parquet(path)
    num=0
    shape=train_img.shape
    for i in range(shape[0]):
        img=train_img.iloc[i]
        image_path=img.image_id+'.png'    
        #create a .png out of the columns and save under theimage_id
        imsave(image_path,img.drop(['image_id']).to_numpy(dtype=np.float64).reshape(HEIGHT, WIDTH).astype(np.float64))
        
        #upload to the bucket
        if num%1000==0:
            upload_blob(BUCKET_NAME,image_path,image_path,printyes=True)
        else:
            upload_blob(BUCKET_NAME,image_path,image_path)
        
        #delete the file in the working directory
        remove(image_path)
        
        num+=1            


# Uploading takes a while so set the following to True only if you havent done this step before. However, if the command was run a second time the images will be overwritten.

# In[ ]:


download_data=False
if download_data:
    for i in range(4):
        print(f'staring with file{i}...')
        upload_bengaliai_data(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet')
        print(f'file {i} finished.')
    
    for i in range(4):
        print(f'staring with file{i}...')
        upload_bengaliai_data(f'/kaggle/input/bengaliai-cv19/test_image_data_{i}.parquet')
        print(f'file {i} finished.')


# ## Pepraring training data

# In order for AutoML to work we need to prepare our data according to [this guideline](https://cloud.google.com/vision/automl/docs/prepare). Therefore, we create a new csv from the train.csv and the test.csv containing the following columns for each datapoint
# 
# 1. 'TRAIN', 'VALIDATION' or 'TEST' determining if the datapoint belong to the training the validation or test set. If this is not set (i.e you start the line with the URI) AutoML will create a test and validation set on its own.
# 2. The google Cloud storage URI
# 3. A comma separated List of the labels that identify how the image is categorized 

# In[ ]:


import csv

BUCKET_LINK='gs://'+BUCKET_NAME

BUCKET_NAME='bengaliai'
BUCKET_LINK='gs://'+BUCKET_NAME
train_labels['uri']=[BUCKET_LINK+'/'+image_id+'.png' for image_id in train_labels['image_id']]
train_labels['g']=['g'+str(num) for num in train_labels['grapheme_root']]
train_labels['v']=['v'+str(num) for num in train_labels['vowel_diacritic']]
train_labels['c']=['c'+str(num) for num in train_labels['consonant_diacritic']]
labels=train_labels.drop(['image_id','grapheme_root','vowel_diacritic','consonant_diacritic','grapheme'],axis=1)
labels.to_csv('all_data.csv',header=False,index=False)


# In[ ]:


all_data=pd.read_csv('all_data.csv')
all_data.head()


# In[ ]:


all_data.shape


# We chose to modify the labels since we have two choices to classify the images
# 1. MultiLabel: Multiple labels are allowed for one example.
# 2. MultiClass: At most one label is allowed per example. 
# 
# In our case we want to tell if and what grapheme_root,vowel_diacritic and consonant_diacritic
# are present in the image. For each of these three components there can be different values 
# (168 different grapheme roots, 11 vowel diacritics and 7 consonant_diacritics). In our model 
# we will view each of these  186 symbols as a class and then predict the presence of the classes 
# in the image with the 'Multilabel' classifier. 
# 
# This might not be the optimal way to express the problem since in every image there can only be 
# one of each component and hence our model will need more examples to learn this apriori known 
# fact (please correct me if I'm wrong with any of my thoughts).
# 
# To sum up our new classes are
# * g0,...,g168, for the grapheme roots
# * v0,...,v11, the vowel diacritics
# * c0,...,c7, corresponding to the consonant diacritics
# 
# It would be better to split the classification task into three part and train on each class of labels seperately.

# ## Create and populate dataset

# We need to create an empty dataset. (again we can consult the [AutoML Tutorial](https://cloud.google.com/vision/automl/docs/tutorial) for the following steps).

# In[ ]:


#set up the AutoMl client

from google.cloud import automl_v1beta1 as automl
#automl_client = automl.AutoMlClient() #not working at the moment

from google.api_core.gapic_v1.client_info import ClientInfo
automl_client = automl.AutoMlClient(client_info=ClientInfo())

display_name='bengaliai_dataset'

# A resource that represents Google Cloud Platform location.
project_location = automl_client.location_path(PROJECT_ID, BUCKET_REGION)


# list the available datasets

# In[ ]:


dataset_names=[]
for dataset in automl_client.list_datasets(project_location):
    dataset_names.append(dataset.name)
    print(dataset.name)


# Choose the dataset and try to load or create a new one:

# In[ ]:


new_dataset=False
try:
    response = automl_client.get_dataset(name=dataset_names[0])
    print('loading successfull.')
except:
    print('couldn\'t get Dataset. Creating new Dataset')
    new_dataset = True
    #Specify the classification type
    #Types:
    #MultiLabel: Multiple labels are allowed for one example.
    #MultiClass: At most one label is allowed per example.
    metadata = automl.types.ImageClassificationDatasetMetadata(classification_type=automl.enums.ClassificationType.MULTILABEL)
    dataset = automl.types.Dataset(display_name=display_name,image_classification_dataset_metadata=metadata)
    response = automl_client.create_dataset(project_location, dataset)
    
    # Create a dataset with the dataset metadata in the region.


# In[ ]:


print("Dataset name: {}".format(response.name))
print("Dataset id: {}".format(response.name.split("/")[-1]))
print("Dataset display name: {}".format(response.display_name))
print("Image classification dataset metadata:")
print("\t{}".format(dataset.image_classification_dataset_metadata))
print("Dataset example count: {}".format(response.example_count))
print("Dataset create time:")
print("\tseconds: {}".format(response.create_time.seconds))
print("\tnanos: {}".format(response.create_time.nanos))


# We need the dataset id of the just created empty dataset.

# In[ ]:


DATASET_ID=response.name.split("/")[-1]


# In[ ]:


# Get the full path of the dataset.=response.name
dataset_full_id = automl_client.dataset_path(PROJECT_ID, 'us-central1', DATASET_ID)


# The import_data method need the dataset.name and the uri to the 'all_data.csv' that we created and uploaded above.

# In[ ]:


all_data_path = 'gs://' + BUCKET_NAME + '/all_data.csv'

input_uris = all_data_path.split(",")
input_config = {"gcs_source": {"input_uris": input_uris}}


import_data=False #set to true if you havent imported
if import_data:
    response=automl_client.import_data(name=response.name,input_config=input_config)

    print("Processing import...")
    # synchronous check of operation status.
    print("Data imported. {}".format(response.result()))


# ## Training

# We can finally create a model and start training in AutoML Vision.

# In[ ]:


# Set model name and model metadata for the image dataset.
TRAIN_BUDGET = 1 # (specified in hours, from 1-100 (int))
MODEL_NAME='bengaliai'


# List the already available models for your dataset

# In[ ]:


models=automl_client.list_models(project_location)
model_names=[]
for md in models:
    model_names.append(md.name)
    print(md.name)


# The first model in this list is the newest.

# In[ ]:


model=None

try:
    model=automl_client.get_model(model_names[0])
    print('loaded the model {}'.format(model.name))
except:
    model_params= {
    "display_name": MODEL_NAME,
    "dataset_id": DATASET_ID,
    "image_classification_model_metadata": {"train_budget": TRAIN_BUDGET}
    if TRAIN_BUDGET
    else {},}
    print('loading model unscucessfull.')
    print('creating new model')
    response=automl_client.create_model(project_location,model_params)
    print("Training operation name: {}".format(response.operation.name))
    print("Training started...")
    
    #wait till training is done
    model=response.result()


# In[ ]:


print('Model name: {}'.format(model.name))
print(print("Model id: {}".format(model.name.split("/")[-1])))

#save the model_id for further use as with the dataset
MODEL_FULL_ID=model.name
MODEL_ID=model.name.split("/")[-1]


# ## Evaluation

# Auto ML provides some scores to evaluate the model. In the [Google Cloud Platform console](https://console.cloud.google.com/vision/dashboard) you also have the option to view model evaluation (precision recall curves etc.) in the browser.

# In[ ]:


print('List of model evaluations:')
num=0#
for evaluation in automl_client.list_model_evaluations(MODEL_FULL_ID, ''):
    if num<=1:
        #take this evaluation and show some metric within        
        response = automl_client.get_model_evaluation(evaluation.name)

        print(u'Model evaluation name: {}'.format(response.name))
        print(u'Model annotation spec id: {}'.format(response.annotation_spec_id))
        print('Create Time:')
        print(u'\tseconds: {}'.format(response.create_time.seconds))
        print(u'\tnanos: {}'.format(response.create_time.nanos / 1e9))
        print(u'Evaluation example count: {}'.format(
            response.evaluated_example_count))
        print('Classification model evaluation metrics: {}'.format(
            response.classification_evaluation_metrics))
        num=num+1
    
    


# In[ ]:


automl_client.list_model_evaluations(MODEL_FULL_ID, '')


# ## Make predictions

# Lastly we look at some predictions of the model (on the test set). To do that we need to deploy the model(this takes a couple of minutes).

# In[ ]:


#response=automl_client.deploy_model(model.name) #uncomment if not deployed before


# In[ ]:


#we need to set up a prediction client 
prediction_client = automl.PredictionServiceClient(client_info=ClientInfo())

#read the file to be predicted
import io
def bengali_make_predict(images):
    """
    Returns a prediction of the grapheme components of the images in 'images'
    -images=pd.dataframe containing the image as rows with first column being the image_id
    """
    for i in range(images.shape[0]):
        #convert the rows to the correct size np.array
        img=images.iloc[i].drop(['image_id']).to_numpy(dtype=np.float64).reshape(HEIGHT, WIDTH).astype(np.float64)
        
        #create a stream as to not save the image in the workspace and pass directly
        imageBytearray=io.BytesIO()
        imsave(imageBytearray,img,format='png')
        
        image=automl.types.Image(image_bytes=imageBytearray.getvalue())
        payload=automl.types.ExamplePayload(image=image)
        
        #define some parameters of the model
        params={'score_threshold': '0.8'}
        
        response=prediction_client.predict(MODEL_FULL_ID,payload, params)
        print('Prediction results:')
        for result in response.payload:
            print(u'Predicted class name: {}'.format(result.display_name))
            print(u'Predicted class score: {}'.format(result.classification.score))


# In[ ]:


test_images=pd.read_parquet('../input/bengaliai-cv19/test_image_data_0.parquet')


# In[ ]:


test_images.shape


# In[ ]:


bengali_make_predict(test_images)


# The predictions still need to be put into the right form for submission to the competition.

# ## The End

# This notebook was written to demonstrate Auto ML from Google. There are many things that could be done more efficiently and the model that was trained above is not very good. Keep this in mind. I still appreciate your comments especially to improve presentation.
