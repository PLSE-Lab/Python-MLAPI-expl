#!/usr/bin/env python
# coding: utf-8

# # Multi-class text classification Problem

# GOAL: Use Supervised Machine Learning Methods and NLP to build a model that classifies new incoming "user complains" into one of the product categories.
#       Target variable: Product
#       Feature: "Consumer complaint narrative"
# 
# https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
# 
# https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb

# ### DATA EXPLORATION

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


# In[ ]:



df = pd.read_csv('../input/Consumer_Complaints.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


#Running the exercise on the first 5000 rows
df = df.head(100_000)


# In[ ]:


df.info()


# ### DATA CLEANING

# In[ ]:


#TO DO: Identify the target and the feature columns


# In[ ]:


#TO DO: Clean the columns (removing missing values)


# In[ ]:


from io import StringIO
col = ['Product', 'Consumer complaint narrative']
df = df[col]
df = df[pd.notnull(df['Consumer complaint narrative'])]
df.columns = ['Product', 'Consumer_complaint_narrative']
df['category_id'] = df['Product'].factorize()[0]
category_id_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Product']].values)
df.head()


# In[ ]:


df.loc[:, ['category_id', 'Product']].drop_duplicates().reset_index(drop=True)


# In[ ]:


#cats_complaints.merge(df.loc[:, ['category_id', 'Product']], how = 'left')


# In[ ]:


#pd.merge(left=cats_complaints, right=df.loc[:, ['category_id', 'Product']], right_on = 'category_id', left_on = 'category_id', how='left')


# In[ ]:


df.loc[:, ['Product', 'category_id']].head(10)


# In[ ]:


df['Consumer_complaint_narrative'].values[3]


# #### Checking for class imbalances

# In[ ]:


df_cats = df.groupby('Product').Consumer_complaint_narrative.count().reset_index()


# In[ ]:


df_cats = df_cats.rename(columns = {'Consumer_complaint_narrative': 'n'})


# In[ ]:


plt.barh(df_cats.sort_values('n')['Product'], df_cats.sort_values('n')['n'])
None


# In[ ]:


df.shape, df.isnull().sum().sum()


# ## BERT

# In[ ]:


from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime


# In[ ]:


get_ipython().system('pip install bert-tensorflow')


# In[ ]:


import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization


# In[ ]:


OUTPUT_DIR = './'#@param {type:"string"}
#@markdown Whether or not to clear/delete the directory and create a new one
DO_DELETE = False #@param {type:"boolean"}
#@markdown Set USE_BUCKET and BUCKET if you want to (optionally) store model output on GCP bucket.
USE_BUCKET = False #@param {type:"boolean"}
BUCKET = None #@param {type:"string"}


# In[ ]:


from tensorflow import keras
import os
import re


# In[ ]:


df.head()


# In[ ]:


df['category_id'].unique()


# In[ ]:


DATA_COLUMN = 'Consumer_complaint_narrative'
LABEL_COLUMN = 'category_id'
# label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]


# In[ ]:


from sklearn.model_selection import train_test_split

train = df.sample(5000)
test = df.sample(5000)


# ## Data Preprocessing

# In[ ]:


train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)


# In[ ]:


BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


# In[ ]:





# In[ ]:


print(tf.__version__)


# In[ ]:


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],tokenization_info["do_lower_case"]])
      
    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


# In[ ]:


print(os.listdir("../.."))


# In[ ]:


tokenizer = create_tokenizer_from_hub_module()


# In[ ]:


MAX_SEQ_LENGTH = 128
# Convert our train and test features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)


# # Creating a model

# In[ ]:


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
    """Creates a classification model."""

    bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
    bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
    bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
    
        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    
        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)


# In[ ]:


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        f1_score = tf.contrib.metrics.f1_score(
            label_ids,
            predicted_labels)
        auc = tf.metrics.auc(
            label_ids,
            predicted_labels)
        recall = tf.metrics.recall(
            label_ids,
            predicted_labels)
        precision = tf.metrics.precision(
            label_ids,
            predicted_labels) 
        true_pos = tf.metrics.true_positives(
            label_ids,
            predicted_labels)
        true_neg = tf.metrics.true_negatives(
            label_ids,
            predicted_labels)   
        false_pos = tf.metrics.false_positives(
            label_ids,
            predicted_labels)  
        false_neg = tf.metrics.false_negatives(
            label_ids,
            predicted_labels)
        return {
            "eval_accuracy": accuracy,
            "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn


# In[ ]:


# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100


# In[ ]:


# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)


# In[ ]:



# Specify outpit directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)


# In[ ]:


model_fn = model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})


# In[ ]:


# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)


# In[ ]:


print(f'Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)


# In[ ]:


test_input_fn = run_classifier.input_fn_builder(
    features=test_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)


# In[ ]:


estimator.evaluate(input_fn=test_input_fn, steps=None)


# In[ ]:


df.loc[:, ['category_id', 'Product']].drop_duplicates().reset_index(drop=True)['Product'].values


# In[ ]:


def getPrediction(in_sentences):
    labels = df.loc[:, ['category_id', 'Product']].drop_duplicates().reset_index(drop=True)['Product'].values
    input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)
    return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]


# In[ ]:


def getPrediction_cat(in_sentences):
    labels = df.loc[:, ['category_id', 'Product']].drop_duplicates().reset_index(drop=True)['Product'].values
    input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)
    return [labels[prediction['labels']] for sentence, prediction in zip(in_sentences, predictions)]


# In[ ]:



pred_sentences = [
  "I am disputing this debt for XXXX XXXX that is currently in collections. I have no knowledge of this account and I have not received any information regarding this account from XXXX or the assigned collection agency.",
  df['Consumer_complaint_narrative'].values[3],
  "The film was creative and surprising",
  "Absolutely fantastic!"
]


# In[ ]:


df['Consumer_complaint_narrative'].values[3]


# In[ ]:


train.shape, test.shape


# In[ ]:


#test['Consumer_complaint_narrative'].values[:100]


# In[ ]:


y_pred = getPrediction_cat(test['Consumer_complaint_narrative'].values)


# In[ ]:


#test['Product'].values


# In[ ]:





# In[ ]:


print(classification_report(test['Product'].values, y_pred, target_names=df['Product'].unique()))


# In[ ]:



predictions = getPrediction(pred_sentences)


# In[ ]:


predictions


# In[ ]:





# In[ ]:


input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in pred_sentences] # here, "" is just a dummy label


# In[ ]:


pred_sentences


# In[ ]:


input_examples


# In[ ]:


input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)

