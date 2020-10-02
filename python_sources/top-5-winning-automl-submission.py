#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# make sure latest version of fastai is installed
#!conda install -c pytorch -c fastai fastai --yes
#!conda install -c anaconda nltk --yes


# In[ ]:


get_ipython().system('pip install pyenchant')
# Enchant needs libenchant to be installed
get_ipython().system(' apt-get update')
get_ipython().system(' apt-get install libenchant-dev -y')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from datetime import datetime

from sklearn.model_selection import train_test_split

from google.cloud import storage
from google.cloud import automl_v1beta1 as automl
# from google.cloud import automl

from automlwrapper import AutoMLWrapper

from fastai.text import *
import spacy
import pandas as pd
import numpy as np
import re
import nltk
from nltk import word_tokenize
from tqdm import tqdm
import enchant
from nltk.metrics import edit_distance


# ## Regex and Spelling functions
# Source: https://github.com/japerk/nltk3-cookbook

# In[ ]:


replacement_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'i\'m', 'i am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are'),
    (r'(\w+)\'d', '\g<1> would'),
]

class RegexpReplacer(object):
    # Replaces regular expression in a text.
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    
    def replace(self, text):
        s = text
        
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        
        return s

class SpellingReplacer(object):
    """ Replaces misspelled words with a likely suggestion based on shortest
    edit distance
    """
    def __init__(self, dict_name='en', max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist
    
    def replace(self, word):
        if self.spell_dict.check(word):
            return word
        
        suggestions = self.spell_dict.suggest(word)
        
        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word


# ## Preparing the data

# In[ ]:


input_path = '/kaggle/input/'
output_path = '/kaggle/working/'


# In[ ]:


# relabel incorrect tweets
ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]
train_orig = pd.read_csv(input_path+'nlp-getting-started/train.csv', encoding = 'latin-1')
incorrect = train_orig[train_orig['id'].isin(ids_with_target_error)]


# In[ ]:


df_train = pd.read_csv(input_path+'disasters-on-social-media/socialmedia-disaster-tweets-DFE.csv', encoding = 'latin-1')
df_train = df_train[['_unit_id','keyword','location','text','choose_one']]
#now grab ids that were identified earlier as mislabelled
incorrect_ids = incorrect.merge(df_train,on=['text'])['_unit_id']
df_train.rename(columns = {'_unit_id': 'id'}, inplace = True)
df_train['target'] = df_train['choose_one'].map({'Relevant': 1, 'Not Relevant': 0})
df_train.drop(['choose_one'], axis = 1, inplace = True)
df_train = df_train[-df_train['target'].isna()]
df_train = df_train.astype({'target': 'int64'})
#reassign incorrect labels
df_train.loc[df_train['id'].isin(incorrect_ids),'target'] = 0
df_train.head()


# In[ ]:


df_train['text'][1]


# In[ ]:


df_test = pd.read_csv(input_path+'nlp-getting-started/test.csv')
df_test.head()


# In[ ]:


# Additional training data
cols = ['target','id','date','flag','user','text']
train_add = pd.read_csv(input_path+'disastertweetsinput/train_clean_add.csv',names=cols,encoding = 'latin-1', skiprows=1)
train_add.head()


# In[ ]:


train_add = df_train[['id','text','target']].append(train_add[['id','text','target']])


# # Cleaning text data

# In[ ]:


def clean_tweet(text) :
    # remove urls
    #text = df.apply(lambda x: re.sub(r'http\S+', '', x))
    text = re.sub(r'http\S+', '', text)

    # replace contractions
    replacer = RegexpReplacer()
    text = replacer.replace(text)

    #split words on - and \
    text = re.sub(r'\b', ' ', text)
    text = re.sub(r'-', ' ', text)

    # replace negations with antonyms

    #nltk.download('punkt')
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)

    # spelling correction
    replacer = SpellingReplacer()
    tokens = [replacer.replace(t) for t in tokens]

    # lemmatize/stemming
    wnl = nltk.WordNetLemmatizer()
    tokens = [wnl.lemmatize(t) for t in tokens]
    porter = nltk.PorterStemmer()
    tokens = [porter.stem(t) for t in tokens]
    # filter insignificant words (using fastai)
    # swap word phrases

    text = ' '.join(tokens)
    return(text)


# In[ ]:


tweets = df_train['text']
tqdm.pandas(desc="Cleaning tweets")
tweets_cleaned = tweets.progress_apply(clean_tweet)
df_train['text_clean'] = tweets_cleaned


# In[ ]:


tweets = df_test['text']
tweets_cleaned = tweets.progress_apply(clean_tweet)
df_test['text_clean'] = tweets_cleaned


# In[ ]:


df = df_test[['text_clean']].append(df_train[['text_clean']])
df = df.drop_duplicates().reset_index()['text_clean']
df.head()


# In[ ]:


#append text_token to df_train and df_test
train = []
tokenizer = Tokenizer()
tok = SpacyTokenizer('en')
for line in tqdm(df_train.text_clean):
    lne = ' '.join(tokenizer.process_text(line, tok))
    train.append(lne)

df_train['text_tokens'] = train
    
test = []
tokenizer = Tokenizer()
tok = SpacyTokenizer('en')
for line in tqdm(df_test.text_clean):
    lne = ' '.join(tokenizer.process_text(line, tok))
    test.append(lne)
    
df_test['text_tokens'] = test


# ## AutoML

# In[ ]:


# Set your own values for these. bucket_name should be the project_id + '-lcm'.
PROJECT_ID = 'kaggle-tweets-0234'
bucket_name = 'kaggle-tweets-0234-lcm'

region = 'us-central1' # Region must be us-central1
dataset_display_name = 'disaster_tweets'
model_display_name = 'disaster_tweets_model1'
storage_client = storage.Client(project=PROJECT_ID)
client = automl.AutoMlClient()


# In[ ]:


# TODO(developer): Uncomment and set the following variables
# project_id = 'YOUR_PROJECT_ID'

# A resource that represents Google Cloud Platform location.
project_location = client.location_path(PROJECT_ID, 'us-central1')
response = client.list_models(project_location, '')

print('List of models:')
for model in response:
    # Display the model information.
    if model.deployment_state ==             automl.enums.Model.DeploymentState.DEPLOYED:
        deployment_state = 'deployed'
    else:
        deployment_state = 'undeployed'

    print(u'Model name: {}'.format(model.name))
    print(u'Model id: {}'.format(model.name.split('/')[-1]))
    print(u'Model display name: {}'.format(model.display_name))
    print(u'Model create time:')
    print(u'\tseconds: {}'.format(model.create_time.seconds))
    print(u'\tnanos: {}'.format(model.create_time.nanos))
    print(u'Model deployment state: {}'.format(deployment_state))


# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#nlp_train_df = pd.read_csv("/kaggle/input/tokenized-disaster-tweets/train_token.csv")
nlp_train_df =  df_train
#nlp_test_df = pd.read_csv("/kaggle/input/tokenized-disaster-tweets/test_token.csv")
nlp_test_df =  df_test
def callback(operation_future):
    result = operation_future.result()


# In[ ]:


nlp_train_df.tail()


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
nlp_train_df[['text_tokens','target']].to_csv('train.csv', index=False, header=False) 


# In[ ]:


nlp_train_df[['id','text_tokens','target']].head()


# In[ ]:


training_gcs_path = 'uploads/kaggle_getstarted/full_train.csv'
#upload_blob(bucket_name, 'train.csv', training_gcs_path)


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
#amw.deploy_model()
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

predictions_df = amw.get_predictions(nlp_test_df, 
                                     input_col_name='text_tokens', 
#                                    ground_truth_col_name='target', # we don't have ground truth in our test set
                                     limit=None, 
                                     threshold=0.5,
                                     verbose=False)


# ## (optional) Undeploy model
# Undeploy the model to stop charges

# In[ ]:


#amw.undeploy_model()


# ## Create submission output

# In[ ]:


predictions_df.head()


# In[ ]:


submission_df = pd.concat([nlp_test_df['id'], predictions_df['class']], axis=1)
submission_df.head()


# In[ ]:


# predictions_df['class'].iloc[:10]
# nlp_test_df['id']


# In[ ]:


submission_df = submission_df.rename(columns={'class':'target'})
submission_df.head()


# ## Submit predictions to the competition!

# In[ ]:


submission_df.to_csv("submission.csv", index=False, header=True)


# In[ ]:


get_ipython().system(' ls -l submission.csv')

