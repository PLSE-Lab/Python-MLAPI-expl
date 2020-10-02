#!/usr/bin/env python
# coding: utf-8

# # Preliminaries

# In[ ]:


# install sent2vec
get_ipython().system('pip install git+https://github.com/epfml/sent2vec')


# Write requirements to file, anytime you run it, in case you have to go back and recover dependencies.
# 
# Latest known such requirements are hosted for each notebook in the companion github repo, and can be pulled down and installed here if needed. Companion github repo is located at https://github.com/azunre/transfer-learning-for-nlp

# In[ ]:


get_ipython().system('pip freeze > kaggle_image_requirements.txt')


# # Download IMDB Movie Review Dataset
# Download IMDB dataset

# In[ ]:


import random
import pandas as pd

## Read-in the reviews and print some basic descriptions of them

get_ipython().system('wget -q "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"')
get_ipython().system('tar xzf aclImdb_v1.tar.gz')


# # Define Tokenization, Stop-word and Punctuation Removal Functions
# Before proceeding, we must decide how many samples to draw from each class. We must also decide the maximum number of tokens per email, and the maximum length of each token. This is done by setting the following overarching hyperparameters

# In[ ]:


Nsamp = 1000 # number of samples to generate in each class - 'spam', 'not spam'
maxtokens = 200 # the maximum number of tokens per document
maxtokenlen = 100 # the maximum length of each token


# **Tokenization**

# In[ ]:


def tokenize(row):
    if row is None or row is '':
        tokens = ""
    else:
        tokens = row.split(" ")[:maxtokens]
    return tokens


# **Use regular expressions to remove unnecessary characters**
# 
# Next, we define a function to remove punctuation marks and other nonword characters (using regular expressions) from the emails with the help of the ubiquitous python regex library. In the same step, we truncate all tokens to hyperparameter maxtokenlen defined above.

# In[ ]:


import re

def reg_expressions(row):
    tokens = []
    try:
        for token in row:
            token = token.lower() # make all characters lower case
            token = re.sub(r'[\W\d]', "", token)
            token = token[:maxtokenlen] # truncate token
            tokens.append(token)
    except:
        token = ""
        tokens.append(token)
    return tokens


# **Stop-word removal**
# 
# Stop-words are also removed. Stop-words are words that are very common in text but offer no useful information that can be used to classify the text. Words such as is, and, the, are are examples of stop-words. The NLTK library contains a list of 127 English stop-words and can be used to filter our tokenized strings.

# In[ ]:


import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')    

# print(stopwords) # see default stopwords
# it may be beneficial to drop negation words from the removal list, as they can change the positive/negative meaning
# of a sentence
# stopwords.remove("no")
# stopwords.remove("nor")
# stopwords.remove("not")


# In[ ]:


def stop_word_removal(row):
    token = [token for token in row if token not in stopwords]
    token = filter(None, token)
    return token


# # Assemble Embedding Vectors
# 
# The following functions are used to extract sent2vec embedding vectors for each review

# In[ ]:


import time
import sent2vec

model = sent2vec.Sent2vecModel()
start=time.time()
model.load_model('../input/sent2vec/wiki_unigrams.bin')
end = time.time()
print("Loading the sent2vec embedding took %d seconds"%(end-start))


# In[ ]:


def assemble_embedding_vectors(data):
    out = None
    for item in data:
        vec = model.embed_sentence(" ".join(item))
        if vec is not None:
            if out is not None:
                out = np.concatenate((out,vec),axis=0)
            else:
                out = vec                                            
        else:
            pass
        
        
    return out


# # Putting It All Together To Assemble Dataset
# 
# Now, putting all the preprocessing steps together we assemble our dataset...

# In[ ]:


import os
import numpy as np

# shuffle raw data first
def unison_shuffle_data(data, header):
    p = np.random.permutation(len(header))
    data = data[p]
    header = np.asarray(header)[p]
    return data, header

# load data in appropriate form
def load_data(path):
    data, sentiments = [], []
    for folder, sentiment in (('neg', 0), ('pos', 1)):
        folder = os.path.join(path, folder)
        for name in os.listdir(folder):
            with open(os.path.join(folder, name), 'r') as reader:
                  text = reader.read()
            text = tokenize(text)
            text = stop_word_removal(text)
            text = reg_expressions(text)
            data.append(text)
            sentiments.append(sentiment)
    data_np = np.array(data)
    data, sentiments = unison_shuffle_data(data_np, sentiments)
    
    return data, sentiments

train_path = os.path.join('aclImdb', 'train')
test_path = os.path.join('aclImdb', 'test')
raw_data, raw_header = load_data(train_path)

print(raw_data.shape)
print(len(raw_header))


# In[ ]:


# Subsample required number of samples
random_indices = np.random.choice(range(len(raw_header)),size=(Nsamp*2,),replace=False)
data_train = raw_data[random_indices]
header = raw_header[random_indices]

print("DEBUG::data_train::")
print(data_train)


# Display sentiments and their frequencies in the dataset, to ensure it is roughly balanced between classes

# In[ ]:


unique_elements, counts_elements = np.unique(header, return_counts=True)
print("Sentiments and their frequencies:")
print(unique_elements)
print(counts_elements)


# **Featurize and Create Labels**

# In[ ]:


EmbeddingVectors = assemble_embedding_vectors(data_train)
print(EmbeddingVectors)


# In[ ]:


data = EmbeddingVectors

idx = int(0.7*data.shape[0])

# 70% of data for training
train_x = data[:idx,:]
train_y = header[:idx]
# # remaining 30% for testing
test_x = data[idx:,:]
test_y = header[idx:] 

print("train_x/train_y list details, to make sure it is of the right form:")
print(len(train_x))
print(train_x)
print(train_y[:5])
print(len(train_y))


# # Logistic Regression Classifier

# In[ ]:


from sklearn.linear_model import LogisticRegression

def fit(train_x,train_y):
    model = LogisticRegression()

    try:
        model.fit(train_x, train_y)
    except:
        pass
    return model

model = fit(train_x,train_y)


# In[ ]:


predicted_labels = model.predict(test_x)
print("DEBUG::The logistic regression predicted labels are::")
print(predicted_labels)


# In[ ]:


from sklearn.metrics import accuracy_score

acc_score = accuracy_score(test_y, predicted_labels)

print("The logistic regression accuracy score is::")
print(acc_score)


# # Random Forests

# In[ ]:


# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=1, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (spam, not spam?)
start_time = time.time()
clf.fit(train_x, train_y)
end_time = time.time()
print("Training the Random Forest Classifier took %3d seconds"%(end_time-start_time))

predicted_labels = clf.predict(test_x)
print("DEBUG::The RF predicted labels are::")
print(predicted_labels)

acc_score = accuracy_score(test_y, predicted_labels)

print("DEBUG::The RF testing accuracy score is::")
print(acc_score)


# In[ ]:


from IPython.display import HTML
def create_download_link(title = "Download file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

#create_download_link(filename='GBMimportances.svg')


# In[ ]:


get_ipython().system('rm -rf aclImdb')
get_ipython().system('rm aclImdb_v1.tar.gz')

