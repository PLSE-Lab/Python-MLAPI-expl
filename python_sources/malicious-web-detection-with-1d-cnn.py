#!/usr/bin/env python
# coding: utf-8

# # MALICIOUS WEB DETECTION WITH 1D CNN (Convolution Neural Network)
# Author: ** Rakha Paleva Kawiswara ** <br>
# 
# Feel free to use this notebook for your research. I'm very appreciate if you give credit and upvotes to this notebook :). Suggestion on this notebook is very expected, Thanks!
# 
# **nb <br>*
# *psst... do not open the url contained in the data, to avoid opening dangerous websites*
# 
# ------------------------------------------------------------------------------------------------------------------------------
# 
# # What is Malicious Website?
# A malicious website is a site that attempts to install malware (a general term for anything that will disrupt computer operation, gather your personal information or, in a worst-case scenario, gain total access to your machine) onto your device. This usually requires some action on your part, however, in the case of a drive-by download, the website will attempt to install software on your computer without asking for permission first. (source: https://us.norton.com/internetsecurity-malware-what-are-malicious-websites.html)
# # Notebook Goals
# This notebook will create a model that can detect malicious websites. Website url is used as a feature and 1D Convolutional Neural Network (CNN) is used as an algorithm for detection malicious websites. Model will be validated by holdout method

# In[ ]:


# install additional library
get_ipython().system('pip install tldextract -q')

# import library
import numpy as np
import pandas as pd
import re
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import seaborn as sns
import gc
import random
import os
import pickle
import tensorflow as tf
from tensorflow.python.util import deprecation
from urllib.parse import urlparse
import tldextract

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models, layers, backend, metrics
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

# set random seed
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

# other setup
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
pd.set_option('max_colwidth', 50)
pio.templates.default = "presentation"
pd.options.plotting.backend = "plotly"
deprecation._PRINT_DEPRECATION_WARNINGS = False


# # Load Data

# First lets load malicious website data

# In[ ]:


# load data
data = pd.read_csv('../input/data.csv')
data.head()


# In this notebook the validation method used is the holdout method. The holdout method is a method that separates training and test data by 80% and 20%

# In[ ]:


val_size = 0.2
train_data, val_data = train_test_split(data, test_size=val_size, stratify=data['label'], random_state=0)
fig = go.Figure([go.Pie(labels=['Train Size', 'Validation Size'], values=[train_data.shape[0], val_data.shape[0]])])
fig.update_layout(title='Train and Validation Size')
fig.show()


# # Data Analysis and Feature Engineering
# Let's do some data analysis to expand our knowledge of this data and do some feature engineering. First we want to find out whether the data is imbalance

# In[ ]:


fig = go.Figure([go.Pie(labels=['Good', 'Bad'], values=data.label.value_counts())])
fig.update_layout(title='Percentage of Class (Good and Bad)')
fig.show()


# As you can see from the plot above, 82% have good labels while 18% have bad labels. 
# 
# Next, lets find out the most used suffix domain, domain and sub domain. We need to extract subdomains, domains and domain suffixes to be able to do the analysis

# In[ ]:


def parsed_url(url):
    # extract subdomain, domain, and domain suffix from url
    # if item == '', fill with '<empty>'
    subdomain, domain, domain_suffix = ('<empty>' if extracted == '' else extracted for extracted in tldextract.extract(url))
    
    return [subdomain, domain, domain_suffix]

def extract_url(data):
    # parsed url
    extract_url_data = [parsed_url(url) for url in data['url']]
    extract_url_data = pd.DataFrame(extract_url_data, columns=['subdomain', 'domain', 'domain_suffix'])
    
    # concat extracted feature with main data
    data = data.reset_index(drop=True)
    data = pd.concat([data, extract_url_data], axis=1)
    
    return data

def get_frequent_group(data, n_group):
    # get the most frequent
    data = data.value_counts().reset_index(name='values')
    
    # scale log base 10
    data['values'] = np.log10(data['values'])
    
    # calculate total values
    # x_column (subdomain / domain / domain_suffix)
    x_column = data.columns[1]
    data['total_values'] = data[x_column].map(data.groupby(x_column)['values'].sum().to_dict())
    
    # get n_group data order by highest values
    data_group = data.sort_values('total_values', ascending=False).iloc[:, 1].unique()[:n_group]
    data = data[data.iloc[:, 1].isin(data_group)]
    data = data.sort_values('total_values', ascending=False)
    
    return data

def plot(data, n_group, title):
    data = get_frequent_group(data, n_group)
    fig = px.bar(data, x=data.columns[1], y='values', color='label')
    fig.update_layout(title=title)
    fig.show()

# extract url
data = extract_url(data)
train_data = extract_url(train_data)
val_data = extract_url(val_data)


# Lets check how many unique Domain, Subdomain and Domain Suffix that we have extracted

# In[ ]:


fig = go.Figure([go.Bar(
    x=['domain', 'Subdomain', 'Domain Suffix'], 
    y = [data.domain.nunique(), data.subdomain.nunique(), data.domain_suffix.nunique()]
)])
fig.show()


# Next, lets plot domain feature

# In[ ]:


plot(
    data=data.groupby('label')['domain'], 
    n_group=20, 
    title='Top 20 Domains Grouped By Labels (Logarithmic Scale)'
)


# Based on the plot above there are interesting things to note, there are some famous domain that don't contain bad labels like linkedin, wikipedia, youtube, amazon, imdb and myspace and there are many famous domain that contains bad labels for example google. Let's take a sample on google domain to see a bad labels

# In[ ]:


data[(data['domain'] == 'google') & (data['label'] == 'bad')].head()


# Maybe some of there urls contain malware, who knows :)
# 
# Lets plot other features like sub domain and domain suffix

# In[ ]:


plot(
    data=data.groupby('label')['subdomain'], 
    n_group=20, 
    title='Top 20 Sub Domains Grouped By Labels (Logarithmic Scale)'
)


# In[ ]:


plot(
    data=data.groupby('label')['domain_suffix'], 
    n_group=20, 
    title='Top 20 Domains Suffix Grouped By Labels (Logarithmic Scale)'
)   


# As you can see from the plot above, most of them contains bad labels even there are some that aren't
# 
# The next step we need to do tokenization on the url so that it can be used as input to the CNN model

# In[ ]:


tokenizer = Tokenizer(filters='', char_level=True, lower=False, oov_token=1)

# fit only on training data
tokenizer.fit_on_texts(train_data['url'])
n_char = len(tokenizer.word_index.keys())

train_seq = tokenizer.texts_to_sequences(train_data['url'])
val_seq = tokenizer.texts_to_sequences(val_data['url'])

print('Before tokenization: ')
print(train_data.iloc[0]['url'])
print('\nAfter tokenization: ')
print(train_seq[0])


# Each url has a different length, therefore padding is needed to equalize each url length. Next step we will do padding on url that we already have tokenize

# In[ ]:


sequence_length = np.array([len(i) for i in train_seq])
sequence_length = np.percentile(sequence_length, 99).astype(int)
print(f'Before padding: \n {train_seq[0]}')
train_seq = pad_sequences(train_seq, padding='post', maxlen=sequence_length)
val_seq = pad_sequences(val_seq, padding='post', maxlen=sequence_length)
print(f'After padding: \n {train_seq[0]}')


# We will also encode subdomain, domain, suffix domains and label into numerical variables

# In[ ]:


unique_value = {}
for feature in ['subdomain', 'domain', 'domain_suffix']:
    # get unique value
    label_index = {label: index for index, label in enumerate(train_data[feature].unique())}
    
    # add unknown label in last index
    label_index['<unknown>'] = list(label_index.values())[-1] + 1
    
    # count unique value
    unique_value[feature] = label_index['<unknown>']
    
    # encode
    train_data.loc[:, feature] = [label_index[val] if val in label_index else label_index['<unknown>'] for val in train_data.loc[:, feature]]
    val_data.loc[:, feature] = [label_index[val] if val in label_index else label_index['<unknown>'] for val in val_data.loc[:, feature]]
    
train_data.head()


# The next step is to encode the target variable (label) to numeric, for example the bad label becomes 1 and the good label becomes 0

# In[ ]:


for data in [train_data, val_data]:
    data.loc[:, 'label'] = [0 if i == 'good' else 1 for i in data.loc[:, 'label']]
    
train_data.head()


# # Create CNN Model

# In[ ]:


def convolution_block(x):
    conv_3_layer = layers.Conv1D(64, 3, padding='same', activation='elu')(x)
    conv_5_layer = layers.Conv1D(64, 5, padding='same', activation='elu')(x)
    conv_layer = layers.concatenate([x, conv_3_layer, conv_5_layer])
    conv_layer = layers.Flatten()(conv_layer)
    return conv_layer

def embedding_block(unique_value, size, name):
    input_layer = layers.Input(shape=(1,), name=name + '_input')
    embedding_layer = layers.Embedding(unique_value, size, input_length=1)(input_layer)
    return input_layer, embedding_layer

def create_model(sequence_length, n_char, unique_value):
    input_layer = []
    
    # sequence input layer
    sequence_input_layer = layers.Input(shape=(sequence_length,), name='url_input')
    input_layer.append(sequence_input_layer)
    
    # convolution block
    char_embedding = layers.Embedding(n_char + 1, 32, input_length=sequence_length)(sequence_input_layer)
    conv_layer = convolution_block(char_embedding)
    
    # entity embedding
    entity_embedding = []
    for key, n in unique_value.items():
        size = 4
        input_l, embedding_l = embedding_block(n + 1, size, key)
        embedding_l = layers.Reshape(target_shape=(size,))(embedding_l)
        input_layer.append(input_l)
        entity_embedding.append(embedding_l)
        
    # concat all layer
    fc_layer = layers.concatenate([conv_layer, *entity_embedding])
    fc_layer = layers.Dropout(rate=0.5)(fc_layer)
    
    # dense layer
    fc_layer = layers.Dense(128, activation='elu')(fc_layer)
    fc_layer = layers.Dropout(rate=0.2)(fc_layer)
    
    # output layer
    output_layer = layers.Dense(1, activation='sigmoid')(fc_layer)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.Precision(), metrics.Recall()])
    return model

# reset session
backend.clear_session()
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

# create model
model = create_model(sequence_length, n_char, unique_value)

# show model architecture
plot_model(model, to_file='model.png')
model_image = mpimg.imread('model.png')
plt.figure(figsize=(75, 75))
plt.imshow(model_image)
plt.show()


# 
# Let me explain to you about the model that has been made. The model received 4 inputs, the first input came from URL that has been done tokenization and padding. Other inputs are subdomains, domains and suffix domains that have been encoded. URL input will pass through embedding layer and convolution layer while other input will pass embedding layer. Then the results from each input will be concatenated.

# # Model Training

# In[ ]:


# create train data
train_x = [train_seq, train_data['subdomain'], train_data['domain'], train_data['domain_suffix']]
train_y = train_data['label'].values

# model training
early_stopping = [EarlyStopping(monitor='val_precision', patience=5, restore_best_weights=True, mode='max')]
history = model.fit(train_x, train_y, batch_size=64, epochs=25, verbose=1, validation_split=0.2, shuffle=True, callbacks=early_stopping)
model.save('model.h5')


# In[ ]:


fig = make_subplots(3, 1, subplot_titles=('loss', 'precision', 'recall'))

for index, key in enumerate(['loss', 'precision', 'recall']):
    # train score
    fig.add_trace(go.Scatter(
        x=list(range(len(history.history[key]))),
        y=history.history[key],
        mode='lines+markers',
        name=key
    ), index + 1, 1)
    
    # val score
    fig.add_trace(go.Scatter(
        x=list(range(len(history.history[f'val_{key}']))),
        y=history.history[f'val_{key}'],
        mode='lines+markers',
        name=f'val {key}'
    ), index + 1, 1)

fig.show()


# # Model Validation

# In[ ]:


val_x = [val_seq, val_data['subdomain'], val_data['domain'], val_data['domain_suffix']]
val_y = val_data['label'].values

val_pred = model.predict(val_x)
val_pred = np.where(val_pred[:, 0] >= 0.5, 1, 0)
print(f'Validation Data:\n{val_data.label.value_counts()}')
print(f'\n\nConfusion Matrix:\n{confusion_matrix(val_y, val_pred)}')
print(f'\n\nClassification Report:\n{classification_report(val_y, val_pred)}')


# # Conclusion
# 
# In conclusion, the model that has been trained has a high precision and recall value but what must be considered is the precision value. The precision value must be high because if it is low then a website that is not malicious has the possibility to be classified as malicious
# 
# 
# Thank you for reading my notebook :)
