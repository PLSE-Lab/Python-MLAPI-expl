#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
import transformers
from transformers import *
from sklearn import metrics
from sklearn.model_selection import KFold


print('Transformers version: ', transformers.__version__)
print('Tensorflow version: ', tf.__version__)


# # Import Data

# In[ ]:


data_dir = '/kaggle/input/nlp-getting-started/'
train_df = pd.read_csv(data_dir+'train.csv')
test_df = pd.read_csv(data_dir+'test.csv')
train_df = train_df.sample(n=len(train_df), random_state=42)
sample_submission = pd.read_csv(data_dir+'sample_submission.csv')
print(train_df['target'].value_counts())
train_df.head(2)


# # Data Prep Functions

# In[ ]:


from nltk.tokenize.treebank import TreebankWordTokenizer
tree_tokenizer = TreebankWordTokenizer()
def get_tree_tokens(x):
    x = tree_tokenizer.tokenize(x)
    x = ' '.join(x)
    return x
train_df.text = train_df.text.apply(get_tree_tokens)
test_df.text = test_df.text.apply(get_tree_tokens)


# In[ ]:


# from: https://www.kaggle.com/utsavnandi/roberta-using-huggingface-tf-implementation
def to_tokens(input_text, tokenizer):
    output = tokenizer.encode_plus(input_text, max_length=90, pad_to_max_length=True)
    return output

def select_field(features, field):
    return [feature[field] for feature in features]

import re
def clean_tweet(tweet):
    # Removing the @
    #tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Removing the URL links
    #tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
    # Keeping only letters
    #tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    # Removing additional whitespaces
    tweet = re.sub(r" +", ' ', tweet)
    return tweet

def preprocess_data(tokenizer, train_df, test_df):
    train_text = train_df['text'].apply(clean_tweet)
    test_text = test_df['text'].apply(clean_tweet)
    train_encoded = train_text.apply(lambda x: to_tokens(x, tokenizer))
    test_encoded = test_text.apply(lambda x: to_tokens(x, tokenizer))

    #create attention masks
    input_ids_train = np.array(select_field(train_encoded, 'input_ids'))
    attention_masks_train = np.array(select_field(train_encoded, 'attention_mask'))

    input_ids_test = np.array(select_field(test_encoded, 'input_ids'))
    attention_masks_test = np.array(select_field(test_encoded, 'attention_mask'))

    # concatonate masks
    train_X = [input_ids_train, attention_masks_train]
    test_X = [input_ids_test, attention_masks_test]
    #OHE target
    train_y = tf.keras.utils.to_categorical(train_df['target'].values.reshape(-1, 1))

    return train_X, train_y, test_X


# # Function to load models

# In[ ]:


# code from https://github.com/huggingface/transformers
# Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
def load_pretrained_model(model_class='bert', model_name='bert-base-cased', task='binary', learning_rate=3e-5, epsilon=1e-8, lower_case=False):
  MODEL_CLASSES = {
    "bert": (BertConfig, TFBertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, TFXLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, TFXLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, TFRobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, TFDistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, TFAlbertForSequenceClassification, AlbertTokenizer),
    #"xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer), No tensorflow version yet
  }
  model_metrics = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
  ]

  
  config_class, model_class, tokenizer_class = MODEL_CLASSES[model_class]

  config = config_class.from_pretrained(model_name, num_labels=2, finetuning_task=task)


  model = model_class.from_pretrained(model_name)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon, clipnorm=1.0)
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  metric = tf.keras.metrics.BinaryAccuracy('accuracy')
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
  #model.summary()

  tokenizer = tokenizer_class.from_pretrained(model_name, lower_case = lower_case)

  return config, model, tokenizer


# # Train Model

# In[ ]:


# load model, process data for model
_, _, tokenizer = load_pretrained_model(model_class='roberta', model_name='roberta-base', learning_rate=2e-5, lower_case=False)
train_X, train_y, test_X = preprocess_data(tokenizer=tokenizer, train_df=train_df, test_df=test_df)


kf = KFold(n_splits=6)
test_preds = []
i = 0
for train_idx, test_idx in kf.split(train_X[0]):
    i+=1
    if i not in [1, 5]: #only do 2 folds to save time
        continue
    train_split_X = [train_X[i][train_idx] for i in range(len(train_X))]
    test_split_X = [train_X[i][test_idx] for i in range(len(train_X))]

    train_split_y = train_y[train_idx]
    test_split_y = train_y[test_idx]
    #create class weights to account for inbalance
    positive = train_df.iloc[train_idx, :].target.value_counts()[0]
    negative = train_df.iloc[train_idx, :].target.value_counts()[1]
    pos_weight = positive / (positive + negative)
    neg_weight = negative / (positive + negative)

    class_weight = [{0:pos_weight, 1:neg_weight}, {0:neg_weight, 1:pos_weight}]

    K.clear_session()
    config, model, tokenizer = load_pretrained_model(model_class='roberta', model_name='roberta-base', learning_rate=2e-5, lower_case=False)

    # fit, test model
    model.fit(train_split_X, train_split_y, batch_size=64, epochs=3, class_weight=class_weight, validation_data=(test_split_X, test_split_y))

    val_preds = model.predict(test_split_X, batch_size=32, verbose=1)
    val_preds = np.argmax(val_preds, axis=1).flatten()
    print(metrics.accuracy_score(train_df.iloc[test_idx, :].target.values, val_preds))

    preds1 = model.predict(test_X, batch_size=32, verbose=1)
    test_preds.append(preds1)


# # Output Predictions

# In[ ]:


test_preds2 = np.average(test_preds, axis=0)
test_preds3 = np.argmax(test_preds2, axis=1).flatten()
sample_submission['target'] = test_preds3
sample_submission['target'].value_counts()
sample_submission.to_csv('new_submission.csv', index=False)

