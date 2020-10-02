#!/usr/bin/env python
# coding: utf-8

# # <font style="color:red;"><center>Unsupervised Learning of <br> Web Content and JavaScript Embedding <br> (Stage-I of Hybrid Deep Learning Model)</center></font>

# ## <font color=blue> Basic Initialisation </font>

# In[ ]:


#Installation of Libraries


# In[ ]:


# Common imports
import pandas as pd
import numpy as np
import time
import warnings
import os
import csv
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input,Dense,Embedding,Bidirectional,LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# to make this notebook's output stable across runs
np.random.seed(42)

#Disabling Warnings
warnings.filterwarnings('ignore')

#Time/CPU Profiling
overall_start_time= time.time()

#TPU Related Setting
from kaggle_datasets import KaggleDatasets
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE


# TPU Hardware Detection

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)


# ### <font color=blue>Loading Dataset </font>

# In[ ]:


# Accessing Data thorugh Google Cloud Storage
GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"


# In[ ]:


#Verifying pathname of dataset before loading - for Kaggle
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename));
        print(os.listdir("../input"))


# In[ ]:


# Load Datasets
def loadDataset(file_name):
    df = pd.read_csv(file_name,engine = 'python')
    return df

start_time= time.time()
df = loadDataset("/kaggle/input/webjavascripttext-size50/WebContent_for_UnsupervisedLearning.csv")
#Removing Unwanted Columns 
df = df[['text']]
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))


# In[ ]:


#Selection lower numbers as of now for fast testing from 361934 rows
#df= df.iloc[:500000,]
print(len(df), 'train examples')


#  ## <font color=blue> Preprocessing the Dataset </font>

# #### Cleaning the Dataset

# In[ ]:


start_time= time.time()
df['text'] = df['text'].str.lower()
print("***Total Time taken --- %s seconds ---***" % (time.time() - start_time))
#Looking for NaN, if any
print(df.isnull().sum())


# In[ ]:


# Removing Stopwords in the text
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))
df


# In[ ]:


#Converting the dataframes into numpy arrays 
text = df['text'].to_numpy()
# text [0] #For checking


# #### Setting of Hyperparameter values for Text Embedding and Encoding

# In[ ]:


#Setting of values of Hyperparameters
vocab_size = 2000
embedding_dim = 50
max_length = 50
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8


# #### Segregating Training and Validation Data

# In[ ]:


# Code to Check the appropriate Batch Size for TPU (divisible by 128) 

# Function to find the largest number smaller 
# than or equal to N that is divisible by k 
def findNum(N, K): 
    rem = N % K 
    if(rem == 0): 
        return N 
    else: 
        return N - rem 

N = 400000
K = 128
print("Largest number smaller than or equal to" + str(N) + "that is divisible by" + str(K) +"is", findNum(N, K))
N = 100000
K = 128
print("Largest number smaller than or equal to" + str(N) + "that is divisible by" + str(K) +"is", findNum(N, K))


# In[ ]:


# Segregating Training and Validation Data
train_size = int(len(text) * training_portion)
train_text = text[0: 400000] # Size Made compatible to TPU
validation_text = text[400000:499968] 
validation_text = validation_text[:198016] # Size Made compatible to TPU
print(train_size)
print(len(train_text))
print(len(validation_text))


# #### Tokenizing and Padding

# In[ ]:


# Tokenizing Words: Training Text
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_text)
word_index = tokenizer.word_index
dict(list(word_index.items())[0:10]) #Checking the first ten in array


# In[ ]:


train_sequences = tokenizer.texts_to_sequences(train_text)
print(train_sequences[10])


# In[ ]:


train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
#Checking 
print(len(train_sequences[0]))
print(len(train_padded[0]))
print(len(train_sequences[1]))
print(len(train_padded[1]))


# In[ ]:


#Checking
print(train_padded[10])


# In[ ]:


# Tokenizing Word: Validation Text
validation_sequences = tokenizer.texts_to_sequences(validation_text)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(len(validation_sequences))
print(validation_padded.shape)


# In[ ]:


#Checking Encoding and Decoding of Tokenizer
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_text(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_text(train_padded[10]))
print('---')
print(train_text[10])


# ## <font color=blue> LSTM AutoEncoder Based Text Coding/Summarization </font>

# In[ ]:


# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


# In[ ]:


# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    #Encoder
    encoder_inputs = Input(shape=(max_length,), name='Encoder-Input')
    emb_layer = Embedding(vocab_size, embedding_dim,input_length = max_length, name='Body-Word-Embedding', mask_zero=False)
    x = emb_layer(encoder_inputs)
    x1 = LSTM(50, return_sequences=True, activation='relu', name='Encoder-LSTM1')(x)
    x2 = LSTM(30, return_sequences=True, activation='relu', name='Encoder-LSTM2')(x1)
    state_h = LSTM(20,activation='relu', name='Encoder-LSTM3')(x2)
    encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
    seq2seq_encoder_out = encoder_model(encoder_inputs)
    encoder_model.summary()
    #Decoder
    decoded = RepeatVector(max_length)(seq2seq_encoder_out)
    decoder_lstm1 = LSTM(20, return_sequences=True, name='Decoder-LSTM-1')
    decoder_lstm1_output = decoder_lstm1(decoded)
    decoder_lstm2 = LSTM(30, return_sequences=True, name='Decoder-LSTM-2')
    decoder_lstm2_output = decoder_lstm2(decoder_lstm1_output)
    decoder_lstm3 = LSTM(50, return_sequences=True, name='Decoder-LSTM-3')
    decoder_lstm3_output = decoder_lstm3(decoder_lstm2_output)
    decoder_dense = Dense(vocab_size, activation='softmax', name='Final-Output-Dense-before')
    decoder_outputs = decoder_dense(decoder_lstm3_output)
    #Combining the Model for training
    seq2seq_Model = Model(encoder_inputs, decoder_outputs)
    seq2seq_Model.summary()
    seq2seq_Model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',metrics=['accuracy'])    


# In[ ]:


# Computing the batch size for max utilisation of TPU
BATCH_SIZE = 16 * tpu_strategy.num_replicas_in_sync
BATCH_SIZE


# In[ ]:


# Training of the Auto Encoder on Combined Model
seq2seq_Model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy')
history = seq2seq_Model.fit(train_padded, np.expand_dims(train_padded, -1),
         batch_size=BATCH_SIZE,
          epochs=100)


# In[ ]:


#Plotting the Accuracy and Loss Achieved in the Unsupervised Model
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  #plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string])
  plt.show()
  
plot_graphs(history, "loss")


# ## <font color=blue> Checking the Text Encoding </font>

# In[ ]:


#Testing with a random text
sentence = 'hello you'
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentence)
seq = tokenizer.texts_to_sequences([sentence])
pad_seq = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
#sentence_vec = encoder_model.predict(pad_seq)
#print(sentence_vec)


# ## <font color=blue> Saving the Model for Stage-II of Hybrid Deep Learning Moded (Supervised Training) </font>

# In[ ]:


# Save the entire model as a SavedModel.
#!mkdir -p saved_model
#encoder_model.save('PretrainedTFModel/1') 


# ## <font color=blue> Re-checking of Stored Model </font>

# In[ ]:


#Re-loading the Saved Model for Checking
#model = tf.keras.models.load_model('PretrainedTFModel/1')

# Check its architecture
#model.summary()


# In[ ]:


#Testing the Embedding from Saved Model
sentence = 'hello'
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentence)
seq = tokenizer.texts_to_sequences([sentence])
pad_seq = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
#sentence_vec = model.predict(pad_seq)
#print(sentence_vec)


# ## <font color=blue> Run Time Profiling Statistics of this Notebook </font>

# In[ ]:


# Total Runtime of this Notebook
print("***Total Time taken --- %s mins ---***" % ((time.time() - overall_start_time)/60))

