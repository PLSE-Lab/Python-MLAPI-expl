#!/usr/bin/env python
# coding: utf-8

# # Acknowledgements
# 
# Original kernels: 
# * https://www.kaggle.com/akensert/bert-base-tf2-0-minimalistic 
# * https://www.kaggle.com/khoongweihao/bert-base-tf2-0-minimalistic-iii

# ### Bert-base TensorFlow 2.0
# 
# This kernel does not explore the data. For that you could check out some of the great EDA kernels: [introduction](https://www.kaggle.com/corochann/google-quest-first-data-introduction), [getting started](https://www.kaggle.com/phoenix9032/get-started-with-your-questions-eda-model-nn) & [another getting started](https://www.kaggle.com/hamditarek/get-started-with-nlp-lda-lsa). This kernel is an example of a TensorFlow 2.0 Bert-base implementation, using TensorFow Hub. <br><br>
# 
# In this kernel, I added a metric function `rho()` which calculates the competition metric (Spearman's correlation coefficient) in order to see the behavior of the model over the epochs.
# 
# The objective is to plot and compare the validation metrics to the training metrics.
# * If both metrics are moving in the same direction, everything is fine.
# * If the validation metric begins to stagnate while the training metric continues to improve, you are probably close to overfitting.
# * If the validation metric is going in the wrong direction, the model is clearly overfitting.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow_hub as hub
import tensorflow as tf
import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import datetime
import gc
import os
from scipy.stats import spearmanr, pearsonr
from math import floor, ceil

np.set_printoptions(suppress=True)


# #### 1. Read data and tokenizer
# 
# Read tokenizer and data, as well as defining the maximum sequence length that will be used for the input to Bert (maximum is usually 512 tokens)

# In[ ]:


PATH = '../input/google-quest-challenge/'
BERT_PATH = '../input/bert-base-from-tfhub/bert_en_uncased_L-12_H-768_A-12'
tokenizer = tokenization.FullTokenizer(BERT_PATH+'/assets/vocab.txt', True)

df_train = pd.read_csv(PATH+'train.csv')
df_test = pd.read_csv(PATH+'test.csv')
df_sub = pd.read_csv(PATH+'sample_submission.csv')
print('train shape =', df_train.shape)
print('test shape =', df_test.shape)

output_categories = list(df_sub.columns[1:])
input_categories = list(df_train.columns[[1,2,5]])
print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)


# #### 2. Preprocessing functions
# 
# These are some functions that will be used to preprocess the raw text data into useable Bert inputs.

# In[ ]:


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def _trim_input(title, question, answer, max_sequence_length, 
                t_max_len=30, q_max_len=239, a_max_len=239):

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
            
            
        if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d" 
                             % (max_sequence_length, (t_new_len+a_new_len+q_new_len+4)))
        
        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]
    
    return t, q, a

def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    
    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows(), total = len(df[columns])):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        t, q, a = _trim_input(t, q, a, max_sequence_length)

        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


# #### 3. Create model
# 
# `CustomCallback()` is a class which inherits from `tf.keras.callbacks.Callback` and will compute and append validation score and validation/test predictions respectively, after each epoch.
# <br><br>
# `bert_model()` contains the actual architecture that will be used to finetune BERT to our dataset. It's simple, just taking the sequence_output of the bert_layer and pass it to an AveragePooling layer and finally to an output layer of 30 units (30 classes that we have to predict)
# <br><br>
# `train_and_predict()` this function will be run to train and obtain predictions
# 
# `compute_spearmanr()` is used to compute the competition metric for the training and validation sets. However, the main function to compute the competition metric will be the `rho()`.  <br><br>
# 
# The following `rho()` metric function uses the spearman correlation and the binary crossentropy. The final metric is the average of all the targets. The basic idea is, if the target has only one unique value, binary crosessentropy will be used (to avoid `nan` in spearman), otherwise, spearman will be used. <br><br>

# In[ ]:


def rho(y_true, y_pred): 
    rhos = tf.constant(0, dtype='float32') 
    for ind in range(30): 
        a = tf.slice(y_true, [0, ind], [-1, 1]) 
        a = tf.reshape(a, [-1]) 
        b = tf.slice(y_pred, [0, ind], [-1, 1]) 
        b = tf.reshape(b, [-1]) 
        rhos = tf.cond(tf.equal(tf.argmax(a), tf.argmin(a)), 
                       lambda: tf.add(rhos, tf.metrics.binary_crossentropy(a, b)), 
                       lambda: tf.add(rhos, tf.py_function(spearmanr, [a, b], Tout=tf.float32))) 
    return tf.divide(rhos, tf.constant(30, 'float32'))


# In[ ]:


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)

class CustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, train_data, valid_data, test_data, batch_size=16, fold=None):
        
        self.train_inputs = train_data[0] #
        self.train_outputs = train_data[1] #

        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.test_inputs = test_data
        
        self.batch_size = batch_size
        self.fold = fold
    
    def on_train_begin(self, logs={}):
        self.train_predictions = [] #
        self.valid_predictions = []
        self.test_predictions = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.train_predictions.append(self.model.predict(self.train_inputs, batch_size=self.batch_size)) #
        rho_train = compute_spearmanr(self.train_outputs, np.average(self.train_predictions, axis=0)) #
        
        self.valid_predictions.append(self.model.predict(self.valid_inputs, batch_size=self.batch_size))
        rho_val = compute_spearmanr(self.valid_outputs, np.average(self.valid_predictions, axis=0))
        print(f"\ntrain rho: %.4f, validation rho: %.4f" % (rho_train, rho_val))
        
        if self.fold is not None:
            self.model.save_weights(f'bert-base-{fold}-{epoch}.h5py')
        
        self.test_predictions.append(
            self.model.predict(self.test_inputs, batch_size=self.batch_size))

def bert_model(output_dim):
    
    input_word_ids = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input(
        (MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')
    
    bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)
    
    _, sequence_output = bert_layer([input_word_ids, input_masks, input_segments])
    
    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(output_dim, activation="sigmoid", name="dense_output")(x)

    model = tf.keras.models.Model(inputs=[input_word_ids, input_masks, input_segments], outputs=out)
    
    return model    
        
def train_and_predict(model, train_data, valid_data, test_data, 
                      learning_rate, epochs, batch_size, loss_function, fold):
        
    custom_callback = CustomCallback(train_data=(train_data[0], train_data[1]), #
                                     valid_data=(valid_data[0], valid_data[1]), 
                                     test_data=test_data,
                                     batch_size=batch_size,
                                     fold=None)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=[rho])
    history = model.fit(x=train_data[0], y=train_data[1], epochs=epochs, 
                        batch_size=batch_size, validation_data=(valid_data[0], valid_data[1]), callbacks=[custom_callback])
    
    return custom_callback, history


# #### 4. Obtain inputs and targets, as well as the indices of the train/validation splits

# In[ ]:


MAX_SEQUENCE_LENGTH = 512
cv = 5

gkf = GroupKFold(n_splits=cv).split(X=df_train.question_body, groups=df_train.question_body) 

outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_arays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)


# #### 5. Training, validation and testing
# 
# Loops over the folds in gkf and trains each fold for the number of epochs --- with a learning rate of 1e-5 and a batch_size. A simple binary crossentropy is used as the objective-/loss-function. 

# In[ ]:


epochs = 6
batch_size = 6
comp_folds = cv+1

custom_callback_histories = []
histories = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    # will actually only do 3 folds (out of 5) to manage < 2h
    if fold < comp_folds:
        K.clear_session()
        model = bert_model(outputs.shape[1])

        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = outputs[train_idx]

        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = outputs[valid_idx]

        # custom_callback_history contains two lists of valid and test preds respectively:
        # [valid_predictions_{fold}, test_predictions_{fold}]
        custom_callback_history, history = train_and_predict(model, 
                                                             train_data=(train_inputs, train_outputs), 
                                                             valid_data=(valid_inputs, valid_outputs),
                                                             test_data=test_inputs,
                                                             learning_rate=3e-5, epochs=epochs, batch_size=batch_size,
                                                             loss_function='binary_crossentropy', fold=fold)
        
        histories.append(history)
        custom_callback_histories.append(custom_callback_history)


# In[ ]:


def metricsplot(histories, metric):
    fig = plt.figure(figsize=(7*2, 4*ceil(len(histories)/2)))
    fig.set_facecolor("#F3F3F3")
    for n in range(len(histories)):
        qx = plt.subplot(ceil(len(histories)/2), 2, n+1)
        plt.plot(list(range(1, len(histories[n].history[metric])+1)), histories[n].history[metric], 'b', linestyle = "dotted", linewidth = 2, 
                 label='Training '+metric)
        plt.plot(list(range(1, len(histories[n].history['val_'+metric])+1)), histories[n].history['val_'+metric], 'b', linewidth = 2, label='Validation '+metric)
        plt.legend(prop = {"size" : 12})
        plt.grid(True, alpha = .15)
        plt.title('Training and validation '+ metric + ' on fold_{%d}' %(n))
        plt.xlabel('Epochs')
        plt.ylabel(metric.title())
        plt.xticks(list(range(1, len(histories[n].history[metric])+1)))


# In[ ]:


metricsplot(histories, 'loss')


# In[ ]:


metricsplot(histories, 'rho')


# #### 6. Process and submit test predictions
# 
# First the test predictions are read from the list of lists of `custom_callback_histories`. Then each test prediction list (in lists) is averaged. Then a mean of the averages is computed to get a single prediction for each data point. Finally, this is saved to `submission.csv`

# In[ ]:


test_predictions = [custom_callback_histories[i].test_predictions for i in range(len(custom_callback_histories))]
test_predictions = [np.average(test_predictions[i], axis=0) for i in range(len(test_predictions))]
test_predictions = np.mean(test_predictions, axis=0)

df_sub.iloc[:, 1:] = test_predictions

df_sub.to_csv('submission.csv', index=False)

