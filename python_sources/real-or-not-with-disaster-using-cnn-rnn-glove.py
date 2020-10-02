#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### **Preparing data for Training**

# In[ ]:


train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv') 
train_df.shape


# In[ ]:


train_df = train_df[['text', 'target']]


# * *I will not take keyword and location their are some keywords which can be useful i will do that later*

# In[ ]:


train_df.head()


# In[ ]:


train_labels = pd.get_dummies(train_df['target'])


# In[ ]:


train_labels


# * *Do some cleaning and Import Packages*

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import os\nimport re\nimport string\n\nimport numpy as np \nfrom string import punctuation\n\nfrom nltk.tokenize import word_tokenize\nfrom nltk.corpus import stopwords\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score, auc, roc_auc_score\n\nfrom tqdm import tqdm\nfrom tqdm import tqdm_notebook\ntqdm.pandas(desc="progress-bar")\n\nimport nltk\nnltk.download(\'punkt\')\nnltk.download(\'stopwords\')\nnltk.download(\'wordnet\')')


# In[ ]:


# Reference : https://www.kaggle.com/sagar7390/nlp-on-disaster-tweets-eda-glove-bert-using-tfhub 

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)


# In[ ]:


train_df['text']=train_df['text'].apply(lambda x : remove_URL(x))
train_df['text']=train_df['text'].apply(lambda x : remove_html(x))
train_df['text']=train_df['text'].apply(lambda x: remove_emoji(x))
train_df['text']=train_df['text'].apply(lambda x : remove_punct(x))


# * **Load Word Vectors in Memory**

# In[ ]:


get_ipython().run_cell_magic('time', '', "def get_coefs(word, *arr):\n    try:\n        return word, np.asarray(arr, dtype='float32')\n    except:\n        return None, None\n    \nembeddings_index = dict(get_coefs(*o.strip().split()) for o in tqdm_notebook(open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt')))\n\nembed_size=300\n\nfor k in tqdm_notebook(list(embeddings_index.keys())):\n    v = embeddings_index[k]\n    try:\n        if v.shape != (embed_size, ):\n            embeddings_index.pop(k)\n    except:\n        pass\n\nif None in embeddings_index:\n  embeddings_index.pop(None)\n  \nvalues = list(embeddings_index.values())\nall_embs = np.stack(values)\n\nemb_mean, emb_std = all_embs.mean(), all_embs.std()")


# ### **Training - RNN and CNN (Glove)**

# * **Import Some packages for training and Tokenize the dataset**

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from keras.preprocessing.text import Tokenizer\nfrom keras.preprocessing.text import text_to_word_sequence\nfrom keras.preprocessing.sequence import pad_sequences\n\nfrom keras.models import Model\nfrom keras.models import Sequential\n\nfrom keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPool2D\nfrom keras.layers import Reshape, Flatten, Dropout, Concatenate\nfrom keras.layers import SpatialDropout1D, concatenate\nfrom keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D\n\nfrom keras.callbacks import Callback\nfrom keras.optimizers import Adam\n\nfrom keras.callbacks import ModelCheckpoint, EarlyStopping\nfrom keras.models import load_model\nfrom keras.utils.vis_utils import plot_model')


# * To use Keras on text data, we firt have to preprocess it. For this, we can use Keras' Tokenizer class. This object takes as argument **num_words** which is the maximum number of words kept after tokenization based on their word frequency.

# In[ ]:


get_ipython().run_cell_magic('time', '', "MAX_NB_WORDS = 80000\ntokenizer = Tokenizer(num_words=MAX_NB_WORDS)\n\ntokenizer.fit_on_texts(train_df['text'])")


# * Once the tokenizer is fitted on the data, we can use it to convert text strings to sequences of numbers.
# 
# * These numbers represent the position of each word in the dictionary (think of it as mapping). 
# 

# * Here's how the tokenizer turns it into a sequence of digits. 

# In[ ]:


get_ipython().run_cell_magic('time', '', "sen = 'Hi Kaggle'\nprint(tokenizer.texts_to_sequences([sen]))")


# * Apply tokenizer in train dataset

# In[ ]:


# %%time
train_sequences = tokenizer.texts_to_sequences(train_df['text'])


# In[ ]:


# find max length of text
def FindMaxLength(lst): 
    maxList = max(lst, key = lambda i: len(i)) 
    maxLength = len(maxList) 
    return maxLength

FindMaxLength(train_sequences)


# * Now the Sentences are mapped to lists of integers. However, we still cannot stack them together in a matrix since they have different lengths.
# Hopefully Keras allows to **pad** sequences with **0s** to a maximum length. We'll set this length to 425.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'MAX_LENGTH = 31\npadded_train_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'padded_train_sequences')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'padded_train_sequences.shape')


# * **Now Start the training**

# In[ ]:


get_ipython().run_cell_magic('time', '', 'word_index = tokenizer.word_index\nnb_words = MAX_NB_WORDS\nembedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))\n\noov = 0\nfor word, i in tqdm_notebook(word_index.items()):\n    if i >= MAX_NB_WORDS: continue\n    embedding_vector = embeddings_index.get(word)\n    if embedding_vector is not None:\n        embedding_matrix[i] = embedding_vector\n    else:\n        oov += 1\n\nprint(oov)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'LABELS = 2\ndef get_rnn_cnn_model_with_glove_embedding():\n    embedding_dim = 300 \n    inp = Input(shape=(MAX_LENGTH, ))\n    x = Embedding(MAX_NB_WORDS, embedding_dim, weights=[embedding_matrix], input_length=MAX_LENGTH, trainable=True)(inp)\n    x = SpatialDropout1D(0.3)(x)\n    x = Bidirectional(GRU(100, return_sequences=True))(x)\n    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)\n    avg_pool = GlobalAveragePooling1D()(x)\n    max_pool = GlobalMaxPooling1D()(x)\n    conc = concatenate([avg_pool, max_pool])\n    outp = Dense(LABELS, activation="sigmoid")(conc)\n    \n    model = Model(inputs=inp, outputs=outp)\n    model.compile(loss=\'binary_crossentropy\',\n                  optimizer=\'adam\',\n                  metrics=[\'accuracy\'])\n    return model\n\nget_rnn_cnn_model_with_embedding = get_rnn_cnn_model_with_glove_embedding()')


# * **Training and Saving the Model**

# In[ ]:


filepath = 'weights-improvement-glove.hdf5'
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

batch_size = 10
epochs = 20

history = get_rnn_cnn_model_with_embedding.fit(x=padded_train_sequences, 
                    y=labels.values, 
                    validation_split = 0.33,
                    batch_size=batch_size, 
                    callbacks=[checkpoint], 
                    epochs=epochs, 
                    verbose=1)


# * Loading the Model

# In[ ]:


get_rnn_cnn_model_with_embedding.load_weights('weights-improvement-glove.hdf5')
get_rnn_cnn_model_with_embedding.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Model is Loaded')


# * **Testing Data**

# In[ ]:


test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
test_df.head()


# In[ ]:


labels_ls = [0, 1]
X_test = test_df['text']
X_test.shape


# In[ ]:


maxlen = 31
token_sen = tokenizer.texts_to_sequences(X_test)
padded_test_sequences = pad_sequences(token_sen, maxlen=maxlen)


# In[ ]:


models = {}
MAX_LENGTH = 31
models['get_rnn_cnn_model_with_embedding'] = {"model": get_rnn_cnn_model_with_embedding,
                                              "process": lambda x: pad_sequences(tokenizer.texts_to_sequences(x), maxlen=MAX_LENGTH)}

y_pred_rnn_cnn_with_glove_embeddings = get_rnn_cnn_model_with_embedding.predict(
    padded_test_sequences, verbose=1, batch_size=2048)


# In[ ]:


result = list((y_pred_rnn_cnn_with_glove_embeddings == y_pred_rnn_cnn_with_glove_embeddings.max(axis=1, keepdims=True)).astype(int))
result_df = pd.DataFrame(result, columns=labels_ls)
result_df['target'] = result_df.idxmax(axis=1)


# In[ ]:


final_df = pd.DataFrame()
final_df['id'] = test_df['id']
final_df['target'] = result_df['target']


# In[ ]:


final_df.to_csv("submission.csv", index=False, header=True)


# In[ ]:




