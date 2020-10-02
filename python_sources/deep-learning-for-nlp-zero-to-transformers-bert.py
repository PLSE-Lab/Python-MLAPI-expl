#!/usr/bin/env python
# coding: utf-8

# # About this Notebook
# 
# NLP is a very hot topic right now and as belived by many experts '2020 is going to be NLP's Year' ,with its ever changing dynamics it is experiencing a boom , same as computer vision once did. Owing to its popularity Kaggle launched two NLP competitions recently and me being a lover of this Hot topic prepared myself to join in my first Kaggle Competition.<br><br>
# As I joined the competitions and since I was a complete beginner with Deep Learning Techniques for NLP, all my enthusiasm took a beating when I saw everyone Using all  kinds of BERT , everything just went over my head,I thought to quit but there is a special thing about Kaggle ,it just hooks you. I thought I have to learn someday , why not now , so I braced myself and sat on the learning curve. I wrote a kernel on the Tweet Sentiment Extraction competition that has now got a gold medal , it can be viewed here : https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model <br><br>
# After 10 days of extensive learning(finishing all the latest NLP approaches) , I am back here to share my leaning , by writing a kernel that starts from the very Basic RNN's to built over , all the way to BERT . I invite you all to come and learn alongside with me and take a step closer towards becoming an NLP expert

# # Contents
# 
# In this Notebook I will start with the very Basics of RNN's and Build all the way to latest deep learning architectures to solve NLP problems. It will cover the Following:
# * Simple RNN's
# * Word Embeddings : Definition and How to get them
# * LSTM's
# * GRU's
# * BI-Directional RNN's
# * Encoder-Decoder Models (Seq2Seq Models)
# * Attention Models
# * Transformers - Attention is all you need
# * BERT
# 
# I will divide every Topic into four subsections:
# * Basic Overview
# * In-Depth Understanding : In this I will attach links of articles and videos to learn about the topic in depth
# * Code-Implementation
# * Code Explanation
# 
# This is a comprehensive kernel and if you follow along till the end , I promise you would learn all the techniques completely
# 
# Note that the aim of this notebook is not to have a High LB score but to present a beginner guide to understand Deep Learning techniques used for NLP. Also after discussing all of these ideas , I will present a starter solution for this competiton

# **<span style="color:Red">This kernel has been a work of more than 10 days If you find my kernel useful and my efforts appreciable, Please Upvote it , it motivates me to write more Quality content**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff


# # Configuring TPU's
# 
# For this version of Notebook we will be using TPU's as we have to built a BERT Model

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')


# We will drop the other columns and approach this problem as a Binary Classification Problem and also we will have our exercise done on a smaller subsection of the dataset(only 12000 data points) to make it easier to train the models

# In[ ]:


train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)


# In[ ]:


train = train.loc[:12000,:]
train.shape


# We will check the maximum number of words that can be present in a comment , this will help us in padding later

# In[ ]:


train['comment_text'].apply(lambda x:len(str(x).split())).max()


# Writing a function for getting auc score for validation

# In[ ]:


def roc_auc(predictions,target):
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels
    '''
    
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


# ### Data Preparation

# In[ ]:


xtrain, xvalid, ytrain, yvalid = train_test_split(train.comment_text.values, train.toxic.values, 
                                                  stratify=train.toxic.values, 
                                                  random_state=42, 
                                                  test_size=0.2, shuffle=True)


# # Before We Begin
# 
# Before we Begin If you are a complete starter with NLP and never worked with text data, I am attaching a few kernels that will serve as a starting point of your journey
# * https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial
# * https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle
# 
# If you want a more basic dataset to practice with here is another kernel which I wrote:
# * https://www.kaggle.com/tanulsingh077/what-s-cooking
# 
# Below are some Resources to get started with basic level Neural Networks, It will help us to easily understand the upcoming parts
# * https://www.youtube.com/watch?v=aircAruvnKk&list=PL_h2yd2CGtBHEKwEH5iqTZH85wLS-eUzv
# * https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PL_h2yd2CGtBHEKwEH5iqTZH85wLS-eUzv&index=2
# * https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PL_h2yd2CGtBHEKwEH5iqTZH85wLS-eUzv&index=3
# * https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PL_h2yd2CGtBHEKwEH5iqTZH85wLS-eUzv&index=4
# 
# For Learning how to visualize test data and what to use view:
# * https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model
# * https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda

# # BERT and Its Implementation on this Competition
# 
# As Promised I am back with Resiurces , to understand about BERT architecture , please follow the contents in the given order :-
# 
# * http://jalammar.github.io/illustrated-bert/ ---> In Depth Understanding of BERT
# 
# After going through the post Above , I guess you must have understood how transformer architecture have been utilized by the current SOTA models . Now these architectures can be used in two ways :<br><br>
# 1) We can use the model for prediction on our problems using the pretrained weights without fine-tuning or training the model for our sepcific tasks
# * EG: http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/ ---> Using Pre-trained BERT without Tuning
# 
# 2) We can fine-tune or train these transformer models for our task by tweaking the already pre-trained weights and training on a much smaller dataset
# * EG:* https://www.youtube.com/watch?v=hinZO--TEk4&t=2933s ---> Tuning BERT For your TASK
# 
# We will be using the first example as a base for our implementation of BERT model using Hugging Face and KERAS , but contrary to first example we will also Fine-Tune our model for our task
# 
# Acknowledgements : https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
# 
# 
# Steps Involved :
# * Data Preparation : Tokenization and encoding of data
# * Configuring TPU's 
# * Building a Function for Model Training and adding an output layer for classification
# * Train the model and get the results

# In[ ]:


# Loading Dependencies
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers

from tokenizers import BertWordPieceTokenizer


# In[ ]:


# LOADING THE DATA

train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')


# Encoder FOr DATA for understanding waht encode batch does read documentation of hugging face tokenizer :
# https://huggingface.co/transformers/main_classes/tokenizer.html here

# In[ ]:


def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)


# In[ ]:


#IMP DATA FOR CONFIG

AUTO = tf.data.experimental.AUTOTUNE


# Configuration
EPOCHS = 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192


# ## Tokenization
# 
# For understanding please refer to hugging face documentation again

# In[ ]:


# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer


# In[ ]:


x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_valid = fast_encode(valid.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)

y_train = train1.toxic.values
y_valid = valid.toxic.values


# In[ ]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)


# In[ ]:


def build_model(transformer, max_len=512):
    """
    function for training the BERT model
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# ## Starting Training
# 
# If you want to use any another model just replace the model name in transformers._____ and use accordingly

# In[ ]:


get_ipython().run_cell_magic('time', '', "with strategy.scope():\n    transformer_layer = (\n        transformers.TFDistilBertModel\n        .from_pretrained('distilbert-base-multilingual-cased')\n    )\n    model = build_model(transformer_layer, max_len=MAX_LEN)\nmodel.summary()")


# In[ ]:


n_steps = x_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)


# In[ ]:


n_steps = x_valid.shape[0] // BATCH_SIZE
train_history_2 = model.fit(
    valid_dataset.repeat(),
    steps_per_epoch=n_steps,
    epochs=EPOCHS*2
)


# In[ ]:


sub['toxic'] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)


# # End Notes
# 
# This was my effort to share my learnings so that everyone can benifit from it.As this community has been very kind to me and helped me in learning all of this , I want to take this forward. I have shared all the resources I used to learn all the stuff .Join me and make these NLP competitions your first ,without being overwhelmed by the shear number of techniques used . It took me 10 days to learn all of this , you can learn it at your pace and dont give in , at the end of all this you will be a different person and it will all be worth it.
# 
# 
# ### I am attaching more resources if you want NLP end to end:
# 
# 1) Books
# 
# * https://d2l.ai/
# * Jason Brownlee's Books
# 
# 2) Courses
# 
# * https://www.coursera.org/learn/nlp-sequence-models/home/welcome
# * Fast.ai NLP Course
# 
# 3) Blogs and websites
# 
# * Machine Learning Mastery
# * https://distill.pub/
# * http://jalammar.github.io/
# 
# **<span style="color:Red">This is subtle effort of contributing towards the community, if it helped you in any way please show a token of love by upvoting**
