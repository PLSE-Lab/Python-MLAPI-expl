#!/usr/bin/env python
# coding: utf-8

# 
# ## Sentiment analysis with Sentence Transformers
# 
# Sentence Transformers: Sentence Embeddings using BERT / RoBERTa / DistilBERT / ALBERT / XLNet with PyTorch
# https://github.com/UKPLab/sentence-transformers
# 
# 
# Idea from this notebook 
# https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta
# 
# Maximilien Roberti trains BERT-like transformers and get **0.7** score in Sentiment analisys competition.
# 
# Here I use pre-trained models to get sentence embeddings and feed them into simple classifier.
# My result is **0.67177** - not so bad and much faster.
# 
# Update:
# Plus two features - num words in sentence and num chars - **0.67569**. It would be 10th result 6 years ago
# 
# 

# ## Libraries and external models.
# 

# In[ ]:


get_ipython().system('pip install -U sentence-transformers')


# In[ ]:


from sentence_transformers import SentenceTransformer

#there are about 10 pretrained models
#roberta-large-nli-stsb-mean-tokens - returns 1024 dimentional vector
#distilbert-base-nli-stsb-mean-tokens - returns 768 dimentional vector

PRETRAINED_MODEL='roberta-large-nli-stsb-mean-tokens'    # 'distilbert-base-nli-stsb-mean-tokens'        
model = SentenceTransformer(PRETRAINED_MODEL)  


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path 
from sklearn import preprocessing
import os
from timeit import default_timer
import tensorflow as tf
import tensorflow.keras as keras
import regex as re
import lightgbm as lgbm


# ## Sentiment Analysis on Movie Reviews
# https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/overview
# 
# The sentiment labels are:
# 
# 0 - negative
# 1 - somewhat negative
# 2 - neutral
# 3 - somewhat positive
# 4 - positive
# 
# 
# Records contains parts of reviews, some of them just one letter 'A'.
# **We will ignore fields PhraseId, SentenceId so result will be far from 100%**

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


DATA_ROOT = Path("..") / "/kaggle/input/sentiment-analysis-on-movie-reviews"
train = pd.read_csv(DATA_ROOT / 'train.tsv.zip', sep="\t")
test = pd.read_csv(DATA_ROOT / 'test.tsv.zip', sep="\t")
print(train.shape,test.shape)
train.head()


# In[ ]:


test.head()


# In[ ]:



#add two simple features - number of chars and words
def add_features (df):
    df['nwords'] = df.Phrase.apply(lambda text: len(re.findall(r'\w+', text)))
    df['nchars'] = df.Phrase.apply(lambda text: len(text))


# In[ ]:


add_features (test)
add_features (train)
test.head()


# Model was loaded. No need to format or tokenize text. All is done inside.
# 
# **Should be done on GPU**

# In[ ]:


#Do it with GPU !!!!!
#on CPU 80 times slower

TRANSFORMER_BATCH=128

def count_embedd (df):
    idx_chunk=list(df.columns).index('Phrase')
    embedd_lst = []
    for index in range (0, df.shape[0], TRANSFORMER_BATCH):
        embedds = model.encode(df.iloc[index:index+TRANSFORMER_BATCH, idx_chunk].values, show_progress_bar=False)
        embedd_lst.append(embedds)
    return np.concatenate(embedd_lst)


# In[ ]:


# sentence embeddings for TRAIN dataset, 1024 dimentions each
start_time = default_timer()
train_embedd = count_embedd(train)
print("Train embeddings: {}: in: {:5.2f}s".format(train_embedd.shape, default_timer() - start_time))


# In[ ]:


# sentence embeddings for TEST dataset, 1024 dimentions each
start_time = default_timer()
test_embedd = count_embedd(test)
print("Test embeddings: {}: in: {:5.2f}s".format(test_embedd.shape, default_timer() - start_time))


# In[ ]:


X_train = np.array(train_embedd)
X_test = np.array(test_embedd)
X_train = np.concatenate((X_train, train[['nchars','nwords']].values), axis=1)
X_test = np.concatenate((X_test, test[['nchars','nwords']].values), axis=1)

X_train.shape, X_test.shape


# In[ ]:


#convert labels into 5-dimentional vector

enc = preprocessing.OneHotEncoder()
label = train['Sentiment'].values.reshape ((-1,1))
enc.fit(label)
y_train = enc.transform(label).toarray()
y_train.shape


# ## Train
# 
# We can use any kind of classifier

# In[ ]:




KERAS_VALIDATION_SPLIT=0.05
KERAS_EPOCHS=10
KERAS_BATCH_SIZE=128

# Create and train Keras model
n_features=X_train.shape[1]
n_labels = y_train.shape[1]

start_time=default_timer()

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(2048, input_dim=n_features, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(n_labels, activation='softmax')
])

LR=0.0001
adam = keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer=adam, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=KERAS_EPOCHS, batch_size=KERAS_BATCH_SIZE, validation_split=KERAS_VALIDATION_SPLIT)

print("Training. Dataset size: {} {:5.2f}s".format(X_train.shape, default_timer() - start_time))


# In[ ]:


y_preds = model.predict(X_test)
y_preds.shape


# In[ ]:


sample_submission = pd.read_csv(DATA_ROOT / 'sampleSubmission.csv')
sample_submission['Sentiment'] = np.argmax(y_preds,axis=1)
sample_submission.to_csv("predictions.csv", index=False)


# ## LightGBM
# 
# only **0.64979** scores

# In[ ]:


params = {
    'objective': 'multiclass',
    'num_class':y_train.shape[1]
    #'metric': 'multi_logloss'
}
lgbm_model = lgbm.LGBMClassifier(objective='multiclass')
lgbm_model.fit(X_train, label)

y_preds = lgbm_model.predict_proba(X_test)

sample_submission = pd.read_csv(DATA_ROOT / 'sampleSubmission.csv')
sample_submission['Sentiment'] = np.argmax(y_preds,axis=1)
sample_submission.to_csv("predictions_lgbm.csv", index=False)


# Compare records in sampleSubmission.csv and results

# In[ ]:


test.tail()


# In[ ]:


sample_submission.tail()


# In[ ]:


from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='predictions_lgbm.csv')

