#!/usr/bin/env python
# coding: utf-8

# # FastAI Text Classification - Beginner Tutorial
# 
# Building on the knowledge & experience gained from the [FastAI Image Classification](https://www.kaggle.com/kkhandekar/fastai-beginner-tutorial) tutorial, we shall attempt to perform a text classification in this notebook.
# 
# And to keep things interesting, we shall be using the [Kick Starter NLP Dataset](https://www.kaggle.com/oscarvilla/kickstarter-nlp).
# 
# Onwards with the scripting ...
# 
# Course of action:
# 
# * Libraries
# * Load, Prep & Explore Data
# * Text Data Pre-Processing
# * Build & Train Model
# * Predictions
# 
# 

# ## Libraries

# In[ ]:


get_ipython().system('pip install contractions -q')


# In[ ]:


#Generic
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re,string,unicodedata
import contractions #import contractions_dict

#FastAI
import fastai
from fastai import *
from fastai.text import * 

#Functional Tool
from functools import partial

#Garbage
import gc

#NLTK
import nltk
from nltk.tokenize.toktok import ToktokTokenizer

#SK Learn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
from sklearn.compose import ColumnTransformer

#Warnings
import warnings
warnings.filterwarnings("ignore")


# ## Load, Prep & Explore Data

# In[ ]:


#Load data
url = '../input/kickstarter-nlp/df_text_eng.csv'
raw_data = pd.read_csv(url, header='infer')


# In[ ]:


#checking the columns
raw_data.columns


# For the purpose of this tutorial, we are only interested in the "blurb" & the "state" columns. So, we shall create a seperate dataframe that will only consist these 2 columns.

# In[ ]:


#creating a seperate dataframe
data = raw_data[['blurb','state']]


# In[ ]:


#inspect the shape of the dataframe
data.shape


# In[ ]:


#Check for null/missing values in the new dataframe
data.isna().sum()


# In[ ]:


#Dropping the records with null/missing values
data = data.dropna()


# In[ ]:


#Checking the records per state
data.groupby('state').size()


# In[ ]:


#Encoding the State label to convert them to numerical values
label_encoder = LabelEncoder() 

#Applying to the dataset
data['state']= label_encoder.fit_transform(data['state']) 


# In[ ]:


#inspect the newly created dataframe
data.head()


# ## Text Data Pre-Processing
# 
# We all have the habit of cleaning our fruits/veggies before eating them, so in the similar way it is always a good practice to clean text data before feeding it to the model. This will stop the Model from spewing incorrect results. 
# 
# In this step we will only focus on cleaning/pre-processing the "blurb" column since the "state" columns is a categorical column.
# 

# In[ ]:


#Remove special characters & retain alphabets
data['blurb'] = data['blurb'].str.replace("[^a-zA-Z]", " ")


# In[ ]:


#Lowering the case
data['blurb'] = data['blurb'].str.lower()

#stripping leading spaces (if any)
data['blurb'] = data['blurb'].str.strip()


# In[ ]:


#removing punctuations
from string import punctuation

def remove_punct(text):
  for punctuations in punctuation:
    text = text.replace(punctuations, '')
  return text

#apply to the dataset
data['blurb'] = data['blurb'].apply(remove_punct)


# In[ ]:


#function to remove macrons & accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

#applying the function on the clean dataset
data['blurb'] = data['blurb'].apply(remove_accented_chars)


# In[ ]:


#Function to expand contractions
def expand_contractions(con_text):
  con_text = contractions.fix(con_text)
  return con_text

#applying the function on the clean dataset
data['blurb'] = data['blurb'].apply(expand_contractions) 


# In[ ]:


#Removing Stopwords
nltk.download('stopwords')

from nltk.corpus import stopwords 
#stop_words = stopwords.words('english')
stopword_list = set(stopwords.words('english'))


# The stopword remover function ingests the text in bite size portions. To achieve this we will have to tokenize (split) our text. Tokenization can be done in 2 ways
# 
# 1. Using the Split function
# 2. Using a Tokenizer function
# 
# We shall implement the tokenization using the second option. NLTK provides a functions for doing just that (check the libraries section above!)
# 

# In[ ]:


#instantiating the tokenizer function
tokenizer = ToktokTokenizer()


# In[ ]:


#function to remove stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

#applying the function
data['blurb_norm'] = data['blurb'].apply(remove_stopwords) 


# In[ ]:


#Dropping the "blurb" column
data = data.drop(['blurb'], axis=1)


# In[ ]:


#Inspect the dataframe after stopword removal
data.head()


# At this stage, our "blurb" column is cleaned and ready to be fed to the Model. We will take a backup of this dataset just incase something goes wrong !

# In[ ]:


#Databack
data_bkup = data.copy()


# ## Build & Train Model
# 
# Before building the model we need to split the dataset into Training & Test/Validation data. We will split the data into 90:10 ratio where 90% of the data will be used for training & remaining 10% for test/validation.
# 

# In[ ]:


#data split
train_data, test_data = train_test_split(data, test_size = 0.1, random_state = 12, stratify=data['state'])


# In[ ]:


train_data.head()


# In[ ]:


#reseting index for test_data
test_data.reset_index(drop=True, inplace=True)

#resting index for train_data
train_data.reset_index(drop=True, inplace=True)


# In[ ]:


#Shape of train & test data
print("Training Data Shape - ",train_data.shape, " Test Data Shape - ", test_data.shape)


# Now, here comes the fun part ..
# 
# ![](https://i2.wp.com/neptune.ai/wp-content/uploads/fastai_logo.png?fit=406%2C194&ssl=1)
# 
# 
# We need to prep our text data for 2 different models i.e. **Language & Classification Model**. This can be done using the FastAI libraries
# 

# In[ ]:


#Language Model
lang_mod = TextLMDataBunch.from_df(train_df= train_data, valid_df=test_data, path='')

#Classification Model
class_mod = TextClasDataBunch.from_df(path='', train_df=train_data, valid_df=test_data, vocab=lang_mod.train_ds.vocab, bs=32)


# Creating a language learner based on the language model (lang_mod) created above.

# In[ ]:


lang_learner = language_model_learner(lang_mod, arch = AWD_LSTM, pretrained = True, drop_mult=0.3)


# In[ ]:


#finding the learning rate for language learner
lang_learner.lr_find()


# In[ ]:


#Plotting the Recorder Plot
lang_learner.recorder.plot()


# And just like in the FastAI Image Classification tutorial we will use the **One Cycle** approach to train our language learner.  The learning rate is chosen based on the plot above.
# The learning rate = 1e-2

# In[ ]:


#Training the language learner model
lang_learner.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))


# Note: Observe that we have achieved an accuracy of ~ 14% , which really bad.

# In[ ]:


#Saving the language learner encoder
lang_learner.save_encoder('fai_langlrn_enc')


# Now, let's use the "class_mod" object created above to build a classifier and then fine-tune our language learner.

# In[ ]:


class_learner = text_classifier_learner(class_mod, drop_mult=0.3, arch = AWD_LSTM, pretrained = True)
class_learner.load_encoder('fai_langlrn_enc')


# In[ ]:


#finding the learning rate of this class_learner
class_learner.lr_find()


# In[ ]:


#Plotting the Recorder Plot for the class learner
class_learner.recorder.plot()


# The learning rate from the above plot is 1e-03

# In[ ]:


#Training the Class Learner Model
class_learner.fit_one_cycle(1, 1e-3, moms=(0.8,0.7))


# As we can observe the accuracy has increased drastically. The current accuracy is at ~ 64% which is strictly OK for the purpose of this tutorial.

# In[ ]:


#saving the Class Learner Model
class_learner.save_encoder('fai_classlrn_enc_tuned')


# In[ ]:


#free memory
gc.collect()


# ## Predictions
# 
# We will now try to get the predictions for the validation set  (test data) from our learner object

# In[ ]:


class_learner.show_results()


# In[ ]:


# predictions
pred, trgt = class_learner.get_preds()


# In[ ]:


#Confusion matrix
prediction = np.argmax(pred, axis = 1)
pd.crosstab (prediction, trgt)


# In[ ]:


#Prediction on Test Dataset
test_dataset = pd.DataFrame({'blurb': test_data['blurb_norm'], 'actual_state' : test_data['state'] })
test_dataset = pd.concat([test_dataset, pd.DataFrame(prediction, columns = ['predicted_state'])], axis=1)

test_dataset.head()


# **Conclusion:** Here we conclude the tutorial for Text Classification using FastAI. As we can clearly observe that the model has an average accuracy due to which not all of the predicted_state is correct. 
# 
# The next step from here is to fine-tune the model to increase the accuracy but that I shall keep it for some other day. Thank you for reading.
